/*
 * \class DPFIsolation
 *
 * Deep ParticleFlow tau isolation using Deep NN.
 *
 * \author Owen Colegrove, UCSB
 */

#include "RecoTauTag/RecoTau/interface/DeepTauBase.h"

namespace {
inline int getPFCandidateIndex(const edm::Handle<pat::PackedCandidateCollection>& pfcands,
                               const reco::CandidatePtr& cptr)
{
    unsigned int pfInd = -1;
    for(unsigned int i = 0; i < pfcands->size(); ++i) {
        pfInd++;
        if(reco::CandidatePtr(pfcands,i) == cptr) {
            pfInd = i;
            break;
        }
    }
    return pfInd;
}
} // anonymous namespace


class DPFIsolation : public deep_tau::DeepTauBase {
public:
    static OutputCollection& GetOutputs()
    {
        static size_t tau_index = 0;
        static OutputCollection outputs = { { "VSall", Output({tau_index}, {}) } };
        return outputs;
    };

    static unsigned GetNumberOfParticles(unsigned graphVersion)
    {
        static const std::map<unsigned, unsigned> nparticles { { 0, 60 }, { 1, 36 } };
        return nparticles.at(graphVersion);
    }

    static unsigned GetNumberOfFeatures(unsigned graphVersion)
    {
        static const std::map<unsigned, unsigned> nfeatures { { 0, 47 }, { 1, 51 } };
        return nfeatures.at(graphVersion);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions)
    {
        edm::ParameterSetDescription desc;
        desc.add<edm::InputTag>("pfcands", edm::InputTag("packedPFCandidates"));
        desc.add<edm::InputTag>("taus", edm::InputTag("slimmedTaus"));
        desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
        desc.add<std::string>("graph_file", "RecoTauTag/TrainingFiles/data/DPFTauId/DPFIsolation_2017v0.pb");

        edm::ParameterSetDescription descWP;
        descWP.add<std::string>("VVVLoose", "0");
        descWP.add<std::string>("VVLoose", "0");
        descWP.add<std::string>("VLoose", "0");
        descWP.add<std::string>("Loose", "0");
        descWP.add<std::string>("Medium", "0");
        descWP.add<std::string>("Tight", "0");
        descWP.add<std::string>("VTight", "0");
        descWP.add<std::string>("VVTight", "0");
        descWP.add<std::string>("VVVTight", "0");
        desc.add<edm::ParameterSetDescription>("VSallWP", descWP);
        descriptions.add("DPFTau2016v0", desc);
    }

    explicit DPFIsolation(const edm::ParameterSet& cfg) :
        DeepTauBase(cfg, GetOutputs()),
        pfcand_token(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("pfcands"))),
        vtx_token(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices")))
    {
        if(graphName.find("v0.pb") != std::string::npos)
            graphVersion = 0;
        else if(graphName.find("v1.pb") != std::string::npos)
            graphVersion = 1;
        else
            throw cms::Exception("DPFIsolation") << "unknown version of the graph file.";

        tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, {1,
            static_cast<int>(GetNumberOfParticles(graphVersion)), static_cast<int>(GetNumberOfFeatures(graphVersion))});
    }

private:
    virtual tensorflow::Tensor GetPredictions(edm::Event& event, const edm::EventSetup& es) override
    {
        event.getByToken(pfcand_token, pfcands);
        event.getByToken(vtx_token, vertices);

        tensorflow::Tensor predictions(tensorflow::DT_FLOAT, { static_cast<int>(taus->size()), 1});

        float pfCandPt, pfCandPz, pfCandPtRel, pfCandPzRel, pfCandDr, pfCandDEta, pfCandDPhi, pfCandEta, pfCandDz,
              pfCandDzErr, pfCandD0, pfCandD0D0, pfCandD0Dz, pfCandD0Dphi, pfCandPuppiWeight,
              pfCandPixHits, pfCandHits, pfCandLostInnerHits, pfCandPdgID, pfCandCharge, pfCandFromPV,
              pfCandVtxQuality, pfCandHighPurityTrk, pfCandTauIndMatch, pfCandDzSig, pfCandD0Sig, pfCandD0Err,
              pfCandPtRelPtRel, pfCandDzDz, pfCandDVx_1, pfCandDVy_1, pfCandDVz_1, pfCandD_1;
        float pvx = !vertices->empty() ? (*vertices)[0].x() : -1;
        float pvy = !vertices->empty() ? (*vertices)[0].y() : -1;
        float pvz = !vertices->empty() ? (*vertices)[0].z() : -1;

        bool pfCandIsBarrel;

        for(size_t tau_index = 0; tau_index < taus->size(); tau_index++) {
            pat::Tau tau = taus->at(tau_index);
            bool isGoodTau = false;
            if(tau.pt() >= 30 && std::abs(tau.eta()) < 2.3 && tau.isTauIDAvailable("againstMuonLoose3") &&
                   tau.isTauIDAvailable("againstElectronVLooseMVA6")) {
                isGoodTau = (tau.tauID("againstElectronVLooseMVA6") && tau.tauID("againstMuonLoose3") );
            }

            if (!isGoodTau) {
                predictions.matrix<float>()(tau_index, 0) = -1;
                continue;
            }

            std::vector<unsigned int> signalCandidateInds;

            for(auto c : tau.signalCands())
                signalCandidateInds.push_back(getPFCandidateIndex(pfcands,c));

            float lepRecoPt = tau.pt();
            float lepRecoPz = std::abs(tau.pz());

            // Use of setZero results in warnings in eigen library during compilation.
            //tensor.flat<float>().setZero();
            const unsigned n_inputs = GetNumberOfParticles(graphVersion) * GetNumberOfFeatures(graphVersion);
            for(unsigned input_idx = 0; input_idx < n_inputs; ++input_idx)
                tensor.flat<float>()(input_idx) = 0;

            unsigned int iPF = 0;
            const unsigned max_iPF = GetNumberOfParticles(graphVersion);

            std::vector<unsigned int> sorted_inds(pfcands->size());
            std::size_t n = 0;
            std::generate(std::begin(sorted_inds), std::end(sorted_inds), [&]{ return n++; });

            std::sort(std::begin(sorted_inds), std::end(sorted_inds),
            [&](int i1, int i2) { return pfcands->at(i1).pt() > pfcands->at(i2).pt(); } );

            for(size_t pf_index = 0; pf_index < pfcands->size() && iPF < max_iPF; pf_index++) {
                pat::PackedCandidate p = pfcands->at(sorted_inds.at(pf_index));
                float deltaR_tau_p =  deltaR(p.p4(),tau.p4());

                if (p.pt() < 0.5) continue;
                if (p.fromPV() < 0) continue;
                if (deltaR_tau_p > 0.5) continue;


                if (p.fromPV() < 1 && p.charge() != 0) continue;
                pfCandPt = p.pt();
                pfCandPtRel = p.pt()/lepRecoPt;

                pfCandDr = deltaR_tau_p;
                pfCandDEta = std::abs(tau.eta() - p.eta());
                pfCandDPhi = std::abs(deltaPhi(tau.phi(), p.phi()));
                pfCandEta = p.eta();
                pfCandIsBarrel = (std::abs(pfCandEta) < 1.4);
                pfCandPz = std::abs(std::sinh(pfCandEta)*pfCandPt);
                pfCandPzRel = std::abs(std::sinh(pfCandEta)*pfCandPt)/lepRecoPz;
                pfCandPdgID = std::abs(p.pdgId());
                pfCandCharge = p.charge();
                pfCandDVx_1 = p.vx() - pvx;
                pfCandDVy_1 = p.vy() - pvy;
                pfCandDVz_1 = p.vz() - pvz;

                pfCandD_1 = std::sqrt(pfCandDVx_1*pfCandDVx_1 + pfCandDVy_1*pfCandDVy_1 + pfCandDVz_1*pfCandDVz_1);

                if (pfCandCharge != 0 and p.hasTrackDetails()){
                    pfCandDz      = p.dz();
                    pfCandDzErr   = p.dzError();
                    pfCandDzSig   = (std::abs(p.dz()) + 0.000001)/(p.dzError() + 0.00001);
                    pfCandD0      = p.dxy();
                    pfCandD0Err   = p.dxyError();
                    pfCandD0Sig   = (std::abs(p.dxy()) + 0.000001)/ (p.dxyError() + 0.00001);
                    pfCandPixHits = p.numberOfPixelHits();
                    pfCandHits    = p.numberOfHits();
                    pfCandLostInnerHits = p.lostInnerHits();
                } else {
                    float disp = 1;
                    int psudorand = p.pt()*1000000;
                    if (psudorand%2 == 0) disp = -1;
                    pfCandDz      = 5*disp;
                    pfCandDzErr   = 0;
                    pfCandDzSig   = 0;
                    pfCandD0      = 5*disp;
                    pfCandD0Err   = 0;
                    pfCandD0Sig   = 0;
                    pfCandPixHits = 0;
                    pfCandHits    = 0;
                    pfCandLostInnerHits = 2.;
                    pfCandDVx_1   = 1;
                    pfCandDVy_1   = 1;
                    pfCandDVz_1   = 1;
                    pfCandD_1     = 1;
                }

                pfCandPuppiWeight = p.puppiWeight();
                pfCandFromPV = p.fromPV();
                pfCandVtxQuality = p.pvAssociationQuality();
                pfCandHighPurityTrk = p.trackHighPurity();
                float pfCandTauIndMatch_temp = 0;

                for (auto i : signalCandidateInds) {
                    if (i == sorted_inds.at(pf_index)) pfCandTauIndMatch_temp = 1;
                }

                pfCandTauIndMatch = pfCandTauIndMatch_temp;
                pfCandPtRelPtRel = pfCandPtRel*pfCandPtRel;
                if (pfCandPt > 500) pfCandPt = 500.;
                pfCandPt = pfCandPt/500.;

                if (pfCandPz > 1000) pfCandPz = 1000.;
                pfCandPz = pfCandPz/1000.;

                if ((pfCandPtRel) > 1 )  pfCandPtRel = 1.;
                if ((pfCandPzRel) > 100 )  pfCandPzRel = 100.;
                pfCandPzRel = pfCandPzRel/100.;
                pfCandDr   = pfCandDr/.5;
                pfCandEta  = pfCandEta/2.75;
                pfCandDEta = pfCandDEta/.5;
                pfCandDPhi = pfCandDPhi/.5;
                pfCandPixHits = pfCandPixHits/7.;
                pfCandHits = pfCandHits/30.;

                if (pfCandPtRelPtRel > 1) pfCandPtRelPtRel = 1;
                pfCandPtRelPtRel = pfCandPtRelPtRel;

                if (pfCandD0 > 5.) pfCandD0 = 5.;
                if (pfCandD0 < -5.) pfCandD0 = -5.;
                pfCandD0 = pfCandD0/5.;

                if (pfCandDz > 5.) pfCandDz = 5.;
                if (pfCandDz < -5.) pfCandDz = -5.;
                pfCandDz = pfCandDz/5.;

                if (pfCandD0Err > 1.) pfCandD0Err = 1.;
                if (pfCandDzErr > 1.) pfCandDzErr = 1.;
                if (pfCandDzSig > 3) pfCandDzSig = 3.;
                pfCandDzSig = pfCandDzSig/3.;

                if (pfCandD0Sig > 1) pfCandD0Sig = 1.;
                pfCandD0D0 = pfCandD0*pfCandD0;
                pfCandDzDz = pfCandDz*pfCandDz;
                pfCandD0Dz = pfCandD0*pfCandDz;
                pfCandD0Dphi = pfCandD0*pfCandDPhi;

                if (pfCandDVx_1 > .05)  pfCandDVx_1 =  .05;
                if (pfCandDVx_1 < -.05) pfCandDVx_1 = -.05;
                pfCandDVx_1 = pfCandDVx_1/.05;

                if (pfCandDVy_1 > 0.05)  pfCandDVy_1 =  0.05;
                if (pfCandDVy_1 < -0.05) pfCandDVy_1 = -0.05;
                pfCandDVy_1 = pfCandDVy_1/0.05;

                if (pfCandDVz_1 > 0.05)  pfCandDVz_1 =  0.05;
                if (pfCandDVz_1 < -0.05) pfCandDVz_1= -0.05;
                pfCandDVz_1 = pfCandDVz_1/0.05;

                if (pfCandD_1 > 0.1)  pfCandD_1 = 0.1;
                if (pfCandD_1 < -0.1) pfCandD_1 = -0.1;
                pfCandD_1 = pfCandD_1/.1;

                if (graphVersion == 0) {
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 0) = pfCandPt;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 1) = pfCandPz;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 2) = pfCandPtRel;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 3) = pfCandPzRel;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 4) = pfCandDr;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 5) = pfCandDEta;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 6) = pfCandDPhi;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 7) = pfCandEta;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 8) = pfCandDz;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 9) = pfCandDzSig;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 10) = pfCandD0;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 11)  = pfCandD0Sig;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 12) = pfCandDzErr;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 13) = pfCandD0Err;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 14) = pfCandD0D0;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 15) = pfCandCharge==0;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 16) = pfCandCharge==1;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 17) = pfCandCharge==-1;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 18) = pfCandPdgID>22;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 19) = pfCandPdgID==22;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 20) = pfCandDzDz;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 21) = pfCandD0Dz;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 22) = pfCandD0Dphi;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 23) = pfCandPtRelPtRel;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 24) = pfCandPixHits;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 25) = pfCandHits;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 26) = pfCandLostInnerHits==-1;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 27) = pfCandLostInnerHits==0;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 28) = pfCandLostInnerHits==1;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 29) = pfCandLostInnerHits==2;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 30) = pfCandPuppiWeight;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 31) = (pfCandVtxQuality == 1);
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 32) = (pfCandVtxQuality == 5);
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 33) = (pfCandVtxQuality == 6);
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 34) = (pfCandVtxQuality == 7);
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 35) = (pfCandFromPV == 1);
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 36) = (pfCandFromPV == 2);
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 37) = (pfCandFromPV == 3);
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 38) = pfCandIsBarrel;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 39) = pfCandHighPurityTrk;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 40) = pfCandPdgID==1;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 41) = pfCandPdgID==2;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 42) = pfCandPdgID==11;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 43) = pfCandPdgID==13;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 44) = pfCandPdgID==130;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 45) = pfCandPdgID==211;
                    tensor.tensor<float,3>()( 0, 60-1-iPF, 46) = pfCandTauIndMatch;
                }

                if (graphVersion == 1) {
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 0) = pfCandPt;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 1) = pfCandPz;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 2) = pfCandPtRel;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 3) = pfCandPzRel;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 4) = pfCandDr;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 5) = pfCandDEta;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 6) = pfCandDPhi;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 7) = pfCandEta;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 8) = pfCandDz;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 9) = pfCandDzSig;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 10) = pfCandD0;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 11) = pfCandD0Sig;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 12) = pfCandDzErr;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 13) = pfCandD0Err;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 14) = pfCandD0D0;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 15) = pfCandCharge==0;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 16) = pfCandCharge==1;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 17) = pfCandCharge==-1;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 18) = pfCandPdgID>22;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 19) = pfCandPdgID==22;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 20) = pfCandDVx_1;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 21) = pfCandDVy_1;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 22) = pfCandDVz_1;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 23) = pfCandD_1;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 24) = pfCandDzDz;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 25) = pfCandD0Dz;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 26) = pfCandD0Dphi;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 27) = pfCandPtRelPtRel;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 28) = pfCandPixHits;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 29) = pfCandHits;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 30) = pfCandLostInnerHits==-1;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 31) = pfCandLostInnerHits==0;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 32) = pfCandLostInnerHits==1;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 33) = pfCandLostInnerHits==2;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 34) = pfCandPuppiWeight;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 35) = (pfCandVtxQuality == 1);
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 36) = (pfCandVtxQuality == 5);
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 37) = (pfCandVtxQuality == 6);
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 38) = (pfCandVtxQuality == 7);
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 39) = (pfCandFromPV == 1);
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 40) = (pfCandFromPV == 2);
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 41) = (pfCandFromPV == 3);
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 42) = pfCandIsBarrel;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 43) = pfCandHighPurityTrk;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 44) = pfCandPdgID==1;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 45) = pfCandPdgID==2;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 46) = pfCandPdgID==11;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 47) = pfCandPdgID==13;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 48) = pfCandPdgID==130;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 49) = pfCandPdgID==211;
                    tensor.tensor<float,3>()( 0, 36-1-iPF, 50) = pfCandTauIndMatch;
                }

                iPF++;
            }

            tensorflow::Status status = session->Run( { {"input_1", tensor} }, {"output_node0"}, {}, &outputs);
            predictions.matrix<float>()(tau_index, 0) = outputs[0].flat<float>()(0);
        }
        return predictions;
    }

private:
    edm::EDGetTokenT<pat::PackedCandidateCollection> pfcand_token;
    edm::EDGetTokenT<reco::VertexCollection>         vtx_token;

    edm::Handle<pat::PackedCandidateCollection>      pfcands;
    edm::Handle<reco::VertexCollection>              vertices;

    unsigned graphVersion;
    tensorflow::Tensor tensor;
    std::vector<tensorflow::Tensor> outputs;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DPFIsolation);

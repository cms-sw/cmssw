#include <string>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/StoppedTrack.h"
#include "DataFormats/PatCandidates/interface/PFIsolation.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/DeDxHitInfo.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "MagneticField/Engine/interface/MagneticField.h"

namespace pat {

    class PATStoppedTrackProducer : public edm::EDProducer {
        public:
            explicit PATStoppedTrackProducer(const edm::ParameterSet&);
            ~PATStoppedTrackProducer();

            virtual void produce(edm::Event&, const edm::EventSetup&);
        
            // compute iso/miniiso
        void GetIsolation(LorentzVector p4, const pat::PackedCandidateCollection* pc, int pc_idx,
                          pat::PFIsolation &iso, pat::PFIsolation &miniiso);

        float GetDeDx(const reco::DeDxHitInfo *hitInfo, bool doPixel, bool doStrip);

        private: 
            
            const edm::EDGetTokenT<pat::PackedCandidateCollection>    pc_;
            const edm::EDGetTokenT<pat::PackedCandidateCollection>    lt_;
            const edm::EDGetTokenT<reco::TrackCollection>    gt_;
            const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > gt2pc_;
            const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > gt2lt_;
            const edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > gt2dedx_;
            const edm::EDGetTokenT<reco::DeDxHitInfoAss> gt2dedxHitInfo_;
            const float pT_cut;  // only save cands with pT>pT_cut
            const float dR_cut;  // isolation radius
            const float dZ_cut;  // save if either from PV or |dz|<dZ_cut
            const std::vector<double> miniIsoParams;
            const float absIso_cut;  // save if ANY of absIso, relIso, or miniRelIso pass the cuts 
            const float relIso_cut;
            const float miniRelIso_cut;

            TrackDetectorAssociator trackAssociator_;
            TrackAssociatorParameters trackAssocParameters_;
    };
}

pat::PATStoppedTrackProducer::PATStoppedTrackProducer(const edm::ParameterSet& iConfig) :
  pc_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
  lt_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("lostTracks"))),
  gt_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("generalTracks"))),
  gt2pc_(consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
  gt2lt_(consumes<edm::Association<pat::PackedCandidateCollection> >(iConfig.getParameter<edm::InputTag>("lostTracks"))),
  gt2dedx_(consumes<edm::ValueMap<reco::DeDxData> >(iConfig.getParameter<edm::InputTag>("dEdxInfo"))),
  gt2dedxHitInfo_(consumes<reco::DeDxHitInfoAss>(iConfig.getParameter<edm::InputTag>("dEdxHitInfo"))),
  pT_cut(iConfig.getParameter<double>("pT_cut")),
  dR_cut(iConfig.getParameter<double>("dR_cut")),
  dZ_cut(iConfig.getParameter<double>("dZ_cut")),
  miniIsoParams(iConfig.getParameter<std::vector<double> >("miniIsoParams")),
  absIso_cut(iConfig.getParameter<double>("absIso_cut")),
  relIso_cut(iConfig.getParameter<double>("relIso_cut")),
  miniRelIso_cut(iConfig.getParameter<double>("miniRelIso_cut"))
{
    // TrackAssociator parameters
    edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
    edm::ConsumesCollector iC = consumesCollector();
    trackAssocParameters_.loadParameters( parameters, iC );

    trackAssociator_.useDefaultPropagator();

    produces< pat::StoppedTrackCollection > ();
}

pat::PATStoppedTrackProducer::~PATStoppedTrackProducer() {}

void pat::PATStoppedTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    edm::Handle<pat::PackedCandidateCollection> pc_h;
    iEvent.getByToken( pc_, pc_h );
    const pat::PackedCandidateCollection *pc = pc_h.product();

    edm::Handle<pat::PackedCandidateCollection> lt_h;
    iEvent.getByToken( lt_, lt_h );

    edm::Handle<reco::TrackCollection> gt_h;
    iEvent.getByToken( gt_, gt_h );
    const reco::TrackCollection *generalTracks = gt_h.product();

    edm::Handle<edm::Association<pat::PackedCandidateCollection> > gt2pc;
    iEvent.getByToken(gt2pc_, gt2pc);

    edm::Handle<edm::Association<pat::PackedCandidateCollection> > gt2lt;
    iEvent.getByToken(gt2lt_, gt2lt);

    edm::Handle<edm::ValueMap<reco::DeDxData> > gt2dedx;
    iEvent.getByToken(gt2dedx_, gt2dedx);

    edm::Handle<reco::DeDxHitInfoAss> gt2dedxHitInfo;
    iEvent.getByToken(gt2dedxHitInfo_, gt2dedxHitInfo);

    edm::ESHandle<HcalChannelQuality> hcalQ_h;
    iSetup.get<HcalChannelQualityRcd>().get("withTopo", hcalQ_h);
    const HcalChannelQuality *hcalQ = hcalQ_h.product();
    
    edm::ESHandle<EcalChannelStatus> ecalS_h;
    iSetup.get<EcalChannelStatusRcd>().get(ecalS_h);
    const EcalChannelStatus *ecalS = ecalS_h.product();

    auto outPtrP = std::make_unique<std::vector<pat::StoppedTrack>>();

    //add general tracks
    for(unsigned int igt=0; igt<generalTracks->size(); igt++){
        const reco::Track &gentk = (*gt_h)[igt];
        reco::TrackRef tkref = reco::TrackRef(gt_h, igt);
        pat::PackedCandidateRef pcref = (*gt2pc)[tkref];
        pat::PackedCandidateRef ltref = (*gt2lt)[tkref];
        bool isInPackedCands = (pcref.isNonnull() && pcref.id()==pc_h.id());
        bool isInLostTracks  = (ltref.isNonnull() && ltref.id()==lt_h.id());

        // apparently this happens when a neutral pfcand is associated with a charged track
        // if(isInPackedCands && isInLostTracks){
        //     std::cout << "THIS SHOULDN'T HAPPEN " << iEvent.eventAuxiliary().event() << " " <<
        //         tkref.key() << " " << tkref.get()->pt() << " " <<
        //         pcref.key() << " " << pcref.get()->pt() << " " <<
        //         ltref.key() << " " << ltref.get()->pt() << std::endl;
        // }

        LorentzVector p4;
        pat::PackedCandidateRef refToCand;
        int pdgId, charge;
        float dz, dxy, dzError, dxyError;
        int pfCandInd; //to avoid counting packedPFCands in their own isolation

        if(isInPackedCands && pcref.get()->charge() != 0){
            p4 = pcref.get()->p4();
            pdgId = pcref.get()->pdgId();
            charge = pcref.get()->charge();
            dz = pcref.get()->dz();
            dxy = pcref.get()->dxy();
            dzError = pcref.get()->dzError();
            dxyError = pcref.get()->dxyError();
            refToCand = pcref;
            pfCandInd = pcref.key();
        }else if(isInLostTracks){
            p4 = ltref.get()->p4();
            pdgId = 0;
            charge = ltref.get()->charge();
            dz = ltref.get()->dz();
            dxy = ltref.get()->dxy();
            dzError = ltref.get()->dzError();
            dxyError = ltref.get()->dxyError();
            refToCand = ltref;
            pfCandInd = -1;
        }else{
            double m = 0.13957018; //assume pion mass
            double E = sqrt(m*m + gentk.p()*gentk.p());
            p4.SetPxPyPzE(gentk.px(), gentk.py(), gentk.pz(), E);
            pdgId = 0;
            charge = gentk.charge();
            dz = gentk.dz();
            dxy = gentk.dxy();
            dzError = gentk.dzError();
            dxyError = gentk.dxyError();
            refToCand = pat::PackedCandidateRef();
            pfCandInd = -1;
        }

        if(p4.pt() < pT_cut)
            continue;
        if(charge == 0)
            continue;

        // get the isolation of the track
        pat::PFIsolation isolationDR03;
        pat::PFIsolation miniIso;
        GetIsolation(p4, pc, pfCandInd, isolationDR03, miniIso);
        
        // isolation cut
        if( !(isolationDR03.chargedHadronIso() < absIso_cut ||
              isolationDR03.chargedHadronIso()/p4.pt() < relIso_cut ||
              miniIso.chargedHadronIso()/p4.pt() < miniRelIso_cut))
            continue;

        if(pfCandInd==-2)
            continue;

        const reco::DeDxHitInfo* hitInfo = (*gt2dedxHitInfo)[tkref].get();
        float dEdxPixel = GetDeDx(hitInfo, true, false);
        // float dEdxPixel = 0.;
        float dEdxStrip = (*gt2dedx)[tkref].dEdx(); // estimated strip dEdx is already stored in AOD
        // float dEdxStrip = 0;

        int trackQuality = gentk.qualityMask();

        // get the associated ecal/hcal detectors
        edm::ESHandle<MagneticField> bField;
        iSetup.get<IdealMagneticFieldRecord>().get(bField);
        FreeTrajectoryState initialState = trajectoryStateTransform::initialFreeState(gentk,&*bField);
        TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, trackAssocParameters_, &initialState);

        // convert to specific hcal DetIds and fill status vectors
        std::vector<HcalDetId> crossedHcalIds;
        std::vector<HcalChannelStatus> crossedHcalStatus;
        for(auto const & did : info.crossedHcalIds){
            crossedHcalIds.push_back(HcalDetId(did));
            crossedHcalStatus.push_back(*(hcalQ->getValues(did.rawId())));
        }
        // fill ecal status vector
        std::vector<EcalChannelStatusCode> crossedEcalStatus;
        for(auto const & did : info.crossedEcalIds){
            crossedEcalStatus.push_back(*(ecalS->find(did.rawId())));
        }

        outPtrP->push_back(pat::StoppedTrack(isolationDR03, miniIso, p4,
                                             charge, pdgId, dz, dxy, dzError, dxyError,
                                             gentk.hitPattern(), dEdxStrip, dEdxPixel, trackQuality, 
                                             info.crossedEcalIds, crossedHcalIds,
                                             crossedEcalStatus, crossedHcalStatus, refToCand));
                                             // std::vector<DetId>(), std::vector<DetId>(), refToCand));

    }

    iEvent.put(std::move(outPtrP));
}


void pat::PATStoppedTrackProducer::GetIsolation(LorentzVector p4, 
                                                const pat::PackedCandidateCollection *pc, int pc_idx,
                                                pat::PFIsolation &iso, pat::PFIsolation &miniiso)
{
        float chiso=0, nhiso=0, phiso=0, puiso=0;   // standard isolation
        float chmiso=0, nhmiso=0, phmiso=0, pumiso=0;  // mini isolation
        float miniDR = std::max(miniIsoParams[0], std::min(miniIsoParams[1], miniIsoParams[2]/p4.pt()));
        for(pat::PackedCandidateCollection::const_iterator pf_it = pc->begin(); pf_it != pc->end(); pf_it++){
            if(pf_it - pc->begin() == pc_idx)  //don't count itself
                continue;
            int id = std::abs(pf_it->pdgId());
            bool fromPV = (pf_it->fromPV()>1 || fabs(pf_it->dz()) < dZ_cut);
            float pt = pf_it->p4().pt();
            float dr = deltaR(p4, pf_it->p4());

            if(dr < dR_cut){
                // charged cands from PV get added to trackIso
                if(id==211 && fromPV)
                    chiso += pt;
                // charged cands not from PV get added to pileup iso
                else if(id==211)
                    puiso += pt;
                // neutral hadron iso
                if(id==130)
                    nhiso += pt;
                // photon iso
                if(id==22)
                    phiso += pt;
            }
            // same for mini isolation
            if(dr < miniDR){
                if(id == 211 && fromPV)
                    chmiso += pt;
                else if(id == 211)
                    pumiso += pt;
                if(id == 130)
                    nhmiso += pt;
                if(id == 22)
                    phmiso += pt;
            }
        }

        iso = pat::PFIsolation(chiso,nhiso,phiso,puiso);
        miniiso = pat::PFIsolation(chmiso,nhmiso,phmiso,pumiso);

}

// get the estimated DeDx in either the pixels or strips (or both)
float pat::PATStoppedTrackProducer::GetDeDx(const reco::DeDxHitInfo *hitInfo, bool doPixel, bool doStrip)
{
    if(hitInfo == NULL){
        return -1;
    }

    std::vector<float> charge_vec;
    for(unsigned int ih=0; ih<hitInfo->size(); ih++){

        // only use pixels. Am i doing this right?
        // if(hitInfo->pixelCluster(ih) == nullptr) continue;
        bool isPixel = (hitInfo->pixelCluster(ih) != nullptr);
        bool isStrip = (hitInfo->stripCluster(ih) != nullptr);

        if(isPixel && !doPixel) continue;
        if(isStrip && !doStrip) continue;

        // probably shouldn't happen
        if(!isPixel && !isStrip) continue;

        // shape selection for strips
        if(isStrip && !DeDxTools::shapeSelection(*(hitInfo->stripCluster(ih))))
            continue;
        
        float Norm=0;       
        if(isPixel)
            Norm = 3.61e-06; //compute the normalization factor to get the energy in MeV/mm
        if(isStrip)
            Norm = 3.61e-06 * 265;
            
        //save the dE/dx in MeV/mm to a vector.  
        charge_vec.push_back(Norm*hitInfo->charge(ih)/hitInfo->pathlength(ih));           
    }

    int size = charge_vec.size();
    float result = 0.0;

    //build the harmonic 2 dE/dx estimator
    float expo = -2;
    for(int i=0; i<size; i++){
        result += pow(charge_vec[i], expo);
    }
    result = (size>0) ? pow(result/size, 1./expo) : 0.0;

    return result;
}

using pat::PATStoppedTrackProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATStoppedTrackProducer);

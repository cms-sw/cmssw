/*
 * \class DPFIsolation
 *
 * Tau identification using Deep NN
 *
 * \author Konstantin Androsov, INFN Pisa
 */

#include <Math/VectorUtil.h>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"
#include "DataFormats/Math/interface/deltaR.h"

inline int getPFCandidateIndex(edm::Handle<pat::PackedCandidateCollection> pfcands, const reco::CandidatePtr cptr){
  unsigned int pfInd = -1;
  for(unsigned int i = 0; i < pfcands->size(); ++i) {
//      const pat::PackedCandidate &pf = (*pfcands)[i];
//    if(pf.pt() < candptMin_) continue;
      pfInd++;
    if(reco::CandidatePtr(pfcands,i) == cptr) {
      pfInd = i;
      break;
    }
  }
  return pfInd;
}

//_____________________________________________________________________________

namespace {

class DPFIsolation : public edm::stream::EDProducer<> {
public:
    using TauType 		= pat::Tau;
    using TauDiscriminator 	= pat::PATTauDiscriminator;
    using TauCollection 	= std::vector<TauType>;
    using TauRef 		= edm::Ref<TauCollection>;
    using TauRefProd 		= edm::RefProd<TauCollection>;
    using ElectronCollection 	= pat::ElectronCollection;
    using MuonCollection 	= pat::MuonCollection;
    using LorentzVectorXYZ 	= ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
    using GraphPtr 		= std::shared_ptr<tensorflow::GraphDef>;

  std::vector<float> calculate(edm::Handle<TauCollection>& taus, 
			       edm::Handle<pat::PackedCandidateCollection>& pfcands, 
			       edm::Handle<reco::VertexCollection>& vertices, 
			       tensorflow::Tensor tensor, 
			       std::vector<tensorflow::Tensor> outputs, 
			       tensorflow::Session* session) {


    std::vector<float> predictions; 
    float pfCandPt, pfCandPz, pfCandPtRel, pfCandPzRel, pfCandDr, pfCandDEta, pfCandDPhi, pfCandEta, pfCandDz,
      pfCandDzErr, pfCandD0, pfCandD0D0, pfCandD0Dz, pfCandD0Dphi, pfCandPuppiWeight,
      pfCandPixHits, pfCandHits, pfCandLostInnerHits, pfCandPdgID, pfCandCharge, pfCandFromPV,
      pfCandVtxQuality, pfCandHighPurityTrk, pfCandTauIndMatch, pfCandDzSig, pfCandD0Sig,pfCandD0Err,pfCandPtRelPtRel,pfCandDzDz,pfCandDVx_1,pfCandDVy_1,pfCandDVz_1,pfCandD_1;
    //std::cout << " Loading the vertices " << std::endl;
    //std::cout << " vertices size = " << vertices->size() << std::endl;
    //std::cout << " (*vertices)[0] = " << (*vertices)[0] << std::endl;
    float pvx = !vertices->empty() ? (*vertices)[0].x() : -1;
    float pvy = !vertices->empty() ? (*vertices)[0].y() : -1;
    float pvz = !vertices->empty() ? (*vertices)[0].z() : -1;
    //std::cout << "vertices loaded " << std::endl;
    //std::cout << " taus->size() = " << taus->size() << std::endl;
    bool pfCandIsBarrel;
    if ( taus->empty() ) return predictions;
    for(size_t tau_index = 0; tau_index < taus->size(); tau_index++) {
      //std::cout << "booting up a tau" << std::endl;

      pat::Tau tau = taus->at(tau_index);
      bool isGoodTau = false;
      if(tau.pt() >= 30 && std::abs(tau.eta()) < 2.3 &&
	 tau.isTauIDAvailable("againstMuonLoose3") &&
	 tau.isTauIDAvailable("againstElectronVLooseMVA6")) {
	
	isGoodTau = (tau.tauID("againstElectronVLooseMVA6") && tau.tauID("againstMuonLoose3") );
      }

      if ( !isGoodTau) {
	predictions.push_back(-1);
	continue;
      }
      
      std::vector<unsigned int> signalCandidateInds;

      for(auto c : tau.signalCands()) 
	signalCandidateInds.push_back(getPFCandidateIndex(pfcands,c));

      float lepRecoPt = tau.pt();
      float lepRecoPz = std::abs(tau.pz());

      tensor.flat<float>().setZero();

      unsigned int iPF = 0;

      std::vector<unsigned int> sorted_inds(pfcands->size());
      std::size_t n(0);
      std::generate(std::begin(sorted_inds), std::end(sorted_inds), [&]{ return n++; });

      std::sort(std::begin(sorted_inds), std::end(sorted_inds),
		[&](int i1, int i2) { return pfcands->at(i1).pt() > pfcands->at(i2).pt(); } );
      //std::cout << "tau booted" << std::endl;

      for(size_t pf_index = 0; pf_index < pfcands->size(); pf_index++) {
	pat::PackedCandidate p = pfcands->at(sorted_inds.at(pf_index));
	float deltaR_tau_p =  deltaR(p.p4(),tau.p4());
	//std::cout << "booting up a pfcanc" << std::endl;

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
	//std::cout << " p.pt() = " << p.pt();
	//std::cout << " p.vx() = " << p.vx();
	//std::cout << " p.vy() = " << p.vy();
	//std::cout << " p.vz() = " << p.vz() << std::endl;

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
	pfCandVtxQuality = p.pvAssociationQuality();//VtxAssocQual();
	pfCandHighPurityTrk = p.trackHighPurity();//HighPurityTrk();
	float pfCandTauIndMatch_temp = 0;
	for (auto i : signalCandidateInds){
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


	//std::cout << "loading tensor  " << std::endl;
	//std::cout << " graphName = " << graphName << std::endl;	  	 
	if (graphName.EndsWith("v0.pb")){// == "RecoTauTag/RecoTau/data/DPFIsolation_2017v0.pb"){
	  //std::cout << " Loading a pfcandidate " << std::endl;
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



	if (graphName.EndsWith("v1.pb")){// == "RecoTauTag/RecoTau/data/DPFIsolation_2017v1.pb"){
	  //std::cout << " Loading a pfcandidate " << std::endl;
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
	if (graphName.EndsWith("v0.pb") and iPF == 60) break;// == "RecoTauTag/RecoTau/data/DPFIsolation_2017v0.pb" and iPF == 60) break;
	if (graphName.EndsWith("v1.pb") and iPF == 36) break;// == "RecoTauTag/RecoTau/data/DPFIsolation_2017v0.pb" and iPF == 60) break;
	
      }
      //std::cout << "tensor loaded" << std::endl;
      //std::cout << " running predictor " << std::endl;

      tensorflow::Status status = session->Run( { {"input_1", tensor}},{"output_node0"}, {}, &outputs);
      //std::cout << " predictor success " << std::endl;
      //float output = outputs[0].scalar<float>()();
      tensorflow::TTypes<float>::Flat output = outputs[0].flat<float>();
      //std::cout << "flatting " << std::endl;
      //std::cout << " output(0) = " << output(0) << std::endl;
      /*if (graphName.EndsWith("v1.pb")){
	for (int iPF =0; iPF < 36; iPF++){
	  for (int iVar = 0; iVar < 51; iVar++){
	    std::cout << tensor.tensor<float,3>()(0,iPF,iVar) << ", "; 
	  }
	  std::cout << std::endl;
	}}
      std::cout << " (*vertices)[0].x() = " << (*vertices)[0].x() << " (*vertices)[0].y() = " << (*vertices)[0].y() << " (*vertices)[0].z() = " << (*vertices)[0].z() << std::endl;
      */
      predictions.push_back(output(0));
    }
    return predictions;
  };

  struct Output {
    std::vector<size_t> num, den;
    Output(std::vector<size_t> _num, std::vector<size_t> _den) : num(_num), den(_den) {}

    std::unique_ptr<TauDiscriminator> get_value(edm::Handle<TauCollection>& taus, std::vector<float> predictions)
    {
      auto output = std::make_unique<TauDiscriminator>(TauRefProd(taus));

      for(size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
	output->setValue(tau_index, predictions.at(tau_index) );
      }
      //std::cout << " successfully got an output = " << std::endl;
      return output;
    }
  };

  using OutputCollection = std::map<std::string, Output>;

  static OutputCollection& GetOutputs()
  {
    static size_t e_index = 0, mu_index = 1, tau_index = 2, jet_index = 3;
    static OutputCollection outputs = {
      { "tauVSe", Output({tau_index}, {e_index, tau_index}) },
      { "tauVSmu", Output({tau_index}, {mu_index, tau_index}) },
      { "tauVSjet", Output({tau_index}, {jet_index, tau_index}) },
      { "tauVSall", Output({tau_index}, {e_index, mu_index, jet_index, tau_index}) }
    };
    return outputs;
  };

public:
  explicit DPFIsolation(const edm::ParameterSet& cfg) :
    taus_token(consumes<TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
    pfcand_token(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("pfcands"))),
    vtx_token(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
    graphName(edm::FileInPath(cfg.getParameter<std::string>("graph_file")).fullPath()),
    graph(tensorflow::loadGraphDef(edm::FileInPath(cfg.getParameter<std::string>("graph_file")).fullPath())),
    session(tensorflow::createSession(graph.get()))
  {
    //std::cout << " The graph file name = " << graphName << std::endl;
    for(auto& output_desc : GetOutputs())
      produces<TauDiscriminator>(output_desc.first);
    //std::cout << " Running a tau id.. " << std::endl;
    if (graphName.EndsWith("v0.pb"))
      tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape( {1, nparticles_v0, nfeatures_v0}));
    if (graphName.EndsWith("v1.pb"))
      tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape( {1, nparticles_v1, nfeatures_v1}));

  };

  virtual ~DPFIsolation() override
  {
    tensorflow::closeSession(session);
  };

  virtual void produce(edm::Event& event, const edm::EventSetup& es) override
  {

    event.getByToken(taus_token, taus);
    event.getByToken(pfcand_token, pfcands);
    event.getByToken(vtx_token, vertices);

    std::vector<float> predictions = DPFIsolation::calculate(taus, pfcands, vertices, tensor, outputs, session);

    for(auto& output_desc : GetOutputs())
      event.put(output_desc.second.get_value(taus,predictions), output_desc.first);
  };

private:
  edm::EDGetTokenT<TauCollection> taus_token;
  edm::EDGetTokenT<pat::PackedCandidateCollection> pfcand_token;
  edm::EDGetTokenT<reco::VertexCollection>         vtx_token;
  
  edm::Handle<pat::TauCollection>                  taus;
  edm::Handle<pat::PackedCandidateCollection>      pfcands;
  edm::Handle<reco::VertexCollection>              vertices;

  TString graphName;
  GraphPtr graph;
  tensorflow::Session* session;
  static const unsigned int nparticles_v0 = 60;
  static const unsigned int nfeatures_v0  = 47;

  static const unsigned int nparticles_v1 = 36;
  static const unsigned int nfeatures_v1  = 51;
  
  tensorflow::Tensor tensor;
  std::vector<tensorflow::Tensor> outputs;

};
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DPFIsolation);

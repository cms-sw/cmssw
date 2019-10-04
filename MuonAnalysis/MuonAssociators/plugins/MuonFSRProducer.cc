// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/Common/interface/Association.h"

//
// class declaration
//

class MuonFSRProducer : public edm::global::EDProducer<> {

public:
  
  explicit MuonFSRProducer(const edm::ParameterSet &iConfig):

    pfcands_ {consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))},
    electrons_ {consumes<pat::ElectronCollection>(iConfig.getParameter<edm::InputTag>("slimmedElectrons"))},
    muons_   {consumes<pat::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))},
    ptCut(iConfig.getParameter<double>("muonPtMin")),
    etaCut(iConfig.getParameter<double>("muonEtaMax")),
    photonPtCut(iConfig.getParameter<double>("photonPtMin"))
	{
      
      produces<std::vector<pat::GenericParticle>>();
      produces<edm::Association<std::vector<pat::GenericParticle>>>();

    }
   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("packedPFCandidates")->setComment("packed pf candidates where to look for photons");
    desc.add<edm::InputTag>("slimmedElectrons")->setComment("electrons to check for footprint");
    desc.add<edm::InputTag>("muons")->setComment("collection of muons to correct for FSR ");
    desc.add<double>("muonPtMin")->setComment("minimum pt of the muon to look for a near photon");
    desc.add<double>("muonEtaMax")->setComment("max eta of the muon to look for a near photon");
    desc.add<double>("photonPtMin")->setComment("minimum photon Pt");

    descriptions.add("MuonFSRProducer", desc);
  } 
  ~MuonFSRProducer() override {}
  
  //static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  double computeRelativeIsolation(const pat::PackedCandidate & photon,
				  const pat::PackedCandidateCollection& pfcands,
				  const double & isoConeMax,
				  const double & isoConeMin) const;

  const edm::EDGetTokenT<pat::PackedCandidateCollection> pfcands_;
  const edm::EDGetTokenT<pat::ElectronCollection> electrons_;
  const edm::EDGetTokenT<pat::MuonCollection> muons_;
  float ptCut;
  float etaCut;
  float photonPtCut;

// ----------member data ---------------------------

  edm::EDGetTokenT<edm::ValueMap<float>> ptFSR_;
  edm::EDGetTokenT<edm::ValueMap<float>> etaFSR_;
  edm::EDGetTokenT<edm::ValueMap<float>> phiFSR_;
  //edm::EDGetTokenT<edm::ValueMap<float>> relIso03FSR_;

};

// //
// // constants, enums and typedefs
// //


// //
// // static data member definitions
// //
void MuonFSRProducer::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

  using namespace std;

  edm::Handle<pat::PackedCandidateCollection> pfcands;
  iEvent.getByToken(pfcands_, pfcands);
  edm::Handle<edm::View<reco::Muon>> muons;
  iEvent.getByToken(muons_, muons);
  edm::Handle<edm::View<pat::Electron>> electrons;
  iEvent.getByToken(electrons_, electrons);

  std::vector<int> muonMapping(muons->size(),-1);
  auto fsrPhotons = std::make_unique<std::vector<pat::GenericParticle>>();
  // loop over all muons
  for (auto muon = muons->begin(); muon != muons->end(); ++muon){

    
    int photonPosition = -1;
    double distance_metric_min = -1;
    // minimum muon pT
    if (muon->pt() < ptCut) continue;
    // maximum muon eta
    if (fabs(muon->eta() > etaCut)) continue;

    // for each muon, loop over all pf cadidates
    for(auto iter_pf = pfcands->begin(); iter_pf != pfcands->end(); iter_pf++){
      auto const & pc = *iter_pf;

      
      // consider only photons
      if (abs(pc.pdgId()) != 22) continue;
      //cout<<"ID pass!"<<endl;

      // minimum pT cut
      if (pc.pt() < photonPtCut) continue;
      //cout<<"pT pass!"<<endl;

      // eta requirements
      // if (fabs(pc.eta()) > 1.4442 and (fabs(pc.eta()) < 1.566)) continue;
      if (fabs(pc.eta()) > 1.4 and (fabs(pc.eta()) < 1.6)) continue;
      if (fabs(pc.eta()) > 2.5) continue;
      //cout<<"ETA pass!"<<endl;

      // 0.0001 < DeltaR(photon,muon) < 0.5 requirement
      double dRPhoMu = deltaR(muon->eta(),muon->phi(),pc.eta(),pc.phi());
      if(dRPhoMu < 0.0001) continue;
      if(dRPhoMu > 0.5) continue;
      //cout<<"DeltaR pass!"<<endl;
	 
      bool skipPhoton = false;
      bool closest = true;

      for (auto othermuon = muons->begin(); othermuon != muons->end(); ++othermuon){
      	if (othermuon->pt() < ptCut or fabs(othermuon->eta() > etaCut)) continue;
	double dRPhoMuOther = deltaR(othermuon->eta(), othermuon->phi(),pc.eta(),pc.phi());
        if (dRPhoMuOther < dRPhoMu) closest = false;
      }
      
      // Check that is not in footprint of an electron
      pat::PackedCandidateRef pfcandRef = pat::PackedCandidateRef(pfcands,iter_pf - pfcands->begin());
      
      for (auto electrons_iter = electrons->begin(); electrons_iter != electrons->end(); ++electrons_iter){
	for(auto itr = electrons_iter->associatedPackedPFCandidates().begin(); itr != electrons_iter->associatedPackedPFCandidates().end(); ++itr)
	  {
	    if(itr->key() == pfcandRef.key()){
	      skipPhoton = true;
	    }
	  }
      }
      
      if(skipPhoton) continue;
      //cout<<"Bremsstrhalung pass!"<<endl;

      if (!closest) continue;
      //cout<<"muon association pass!"<<endl;

      // use only isolated photons (very loose prelection can be tightened on analysis level)
      float photon_relIso03 = computeRelativeIsolation(pc,*pfcands,0.3,0.0001);
      if(photon_relIso03 > 0.8) continue;
      fsrPhotons->push_back(pat::GenericParticle(pc));
      fsrPhotons->back().addUserFloat("relIso03",photon_relIso03); // isolation, no CHS
      fsrPhotons->back().addUserCand("associatedMuon",reco::CandidatePtr(muons,muon-muons->begin()));
      double metric = deltaR(muon->eta(),muon->phi(),pc.eta(),pc.phi())/(pc.pt()*pc.pt());
      fsrPhotons->back().addUserFloat("dROverEt2",metric); // dR/et2 to the closest muon

      // FSR photon defined as the one with minimum value of DeltaR/Et^2
      if(photonPosition == -1 or metric < distance_metric_min){
      	distance_metric_min = metric;
      	photonPosition = fsrPhotons->size();
      }
       
    }
    muonMapping[muon-muons->begin()] = photonPosition; 
  }

  edm::OrphanHandle<std::vector<pat::GenericParticle>> oh = iEvent.put(std::move(fsrPhotons));
  auto muon2photon = std::make_unique<edm::Association<std::vector<pat::GenericParticle>>>(oh);
  edm::Association<std::vector<pat::GenericParticle>>::Filler muon2photonFiller(*muon2photon);
  muon2photonFiller.insert(muons, muonMapping.begin(), muonMapping.end());
  muon2photonFiller.fill();
  iEvent.put(std::move(muon2photon));

}

double MuonFSRProducer::computeRelativeIsolation(const pat::PackedCandidate & photon,
						 const pat::PackedCandidateCollection& pfcands,
						 const double & isoConeMax,
						 const double & isoConeMin) const{

  double ptsum = 0;

  for(auto pfcand : pfcands){
    
    // Isolation cone requirement
    double dRIsoCone = deltaR(photon.eta(),photon.phi(),pfcand.eta(),pfcand.phi());
    if(dRIsoCone > isoConeMax) continue;
    if(dRIsoCone < isoConeMin) continue;
    
    if (pfcand.charge() != 0 && abs(pfcand.pdgId()) == 211 && pfcand.pt() > 0.2) {
      if (dRIsoCone > 0.0001)
        ptsum += pfcand.pt();
    } else if (pfcand.charge() == 0 &&
               (abs(pfcand.pdgId()) == 22 || abs(pfcand.pdgId()) == 130) && pfcand.pt() > 0.5) {
      if (dRIsoCone > 0.01)
        ptsum += pfcand.pt();
    }
  }

  return ptsum/photon.pt();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonFSRProducer);


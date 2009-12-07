// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"



#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"



using namespace std;
using namespace edm;
using namespace reco;


class MuonFilter : public EDFilter {
 public:
  explicit MuonFilter(const edm::ParameterSet& pset);
  ~MuonFilter();
  virtual bool filter(edm::Event& event, const edm::EventSetup& eventSetup);
       
protected:
       
private: 

  InputTag theMuonTag;
  InputTag theCaloMuonTag;
  InputTag l1MuonLabel;

  bool selectL1Trigger;

  bool selectOnDTHits;
  bool selectOnRPCHits;

  bool selectMuons;
  bool selectCaloMuons;

};


MuonFilter::MuonFilter(const ParameterSet& pset){
  // input tags

  theMuonTag         =  pset.getParameter<edm::InputTag>("muonsLabel");
  theCaloMuonTag     =  pset.getParameter<edm::InputTag>("caloMuonsLabel");

  selectL1Trigger = pset.getParameter<bool>("selectL1Trigger");

  selectOnDTHits = pset.getParameter<bool>("selectOnDTHits");
  selectOnRPCHits = pset.getParameter<bool>("selectOnRPCHits");

  selectMuons      =  pset.getParameter<bool>("selectMuons");
  selectCaloMuons  =  pset.getParameter<bool>("selectCaloMuons");
}

MuonFilter::~MuonFilter(){
}

bool MuonFilter::filter(edm::Event& event, const edm::EventSetup& eventSetup){

 const std::string metname = "Muon|RecoMuon|L2MuonSeedGenerator";

 bool accept=false;
 
 // Check L1 Trigger
 if(selectL1Trigger){
   Handle<l1extra::L1MuonParticleCollection> muL1;
   event.getByLabel(l1MuonLabel, muL1);
   LogTrace(metname) << "Number of L1 muons " << muL1->size() << endl;
   accept |= (muL1->size()>0);
 }
    
  // Check DT hits
  if(selectOnDTHits){}

  // Check RPC hits
  if(selectOnRPCHits){}

  // Check muons
  if(selectMuons){
    Handle<reco::MuonCollection> muons;
    event.getByLabel(theMuonTag,muons);
    LogTrace(metname) << "Number of muons " << muons->size() << endl;
    accept |= (muons->size()>0);
  }
    
  // Check caloMuons
  if(selectCaloMuons){
    Handle<reco::CaloMuonCollection> caloMuons;
    event.getByLabel(theCaloMuonTag,caloMuons);
    LogTrace(metname) << "Number of caloMuons " << caloMuons->size() << endl;
    accept |= (caloMuons->size()>0);
  }

  return accept;
}

DEFINE_FWK_MODULE(MuonFilter);

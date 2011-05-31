/** \class MuonProducer
 *  No description available.
 *
 *  $Date: 2010/03/25 14:08:49 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/MuonIdentification/plugins/MuonProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#ifndef dout
#define dout if(debug_) std::cout
#endif

using std::endl;

typedef std::map<reco::MuonRef, reco::Candidate::LorentzVector> MuToPFMap;

/// Constructor
MuonProducer::MuonProducer(const edm::ParameterSet& pSet):debug_(pSet.getUntrackedParameter<bool>("ActivateDebug",false)){

  theMuonsCollectionLabel = pSet.getParameter<edm::InputTag>("InputMuons");
  thePFCandLabel = pSet.getParameter<edm::InputTag>("PFCandidates");

  // FIXME: need to update the asso map too!!!!!!
  //  setAlias(pSet.getParameter<std::string>("@module_label"));
  //  produces<reco::MuonCollection>().setBranchAlias(theAlias + "s");
  produces<reco::MuonCollection>();
//   produces<reco::MuonTimeExtraMap>("combined");
//   produces<reco::MuonTimeExtraMap>("dt");
//   produces<reco::MuonTimeExtraMap>("csc");

//  if (fillIsolation_ && writeIsoDeposits_){
//      trackDepositName_ = iConfig.getParameter<std::string>("trackDepositName");
//      produces<reco::IsoDepositMap>(trackDepositName_);
//      ecalDepositName_ = iConfig.getParameter<std::string>("ecalDepositName");
//      produces<reco::IsoDepositMap>(ecalDepositName_);
//      hcalDepositName_ = iConfig.getParameter<std::string>("hcalDepositName");
//      produces<reco::IsoDepositMap>(hcalDepositName_);
//      hoDepositName_ = iConfig.getParameter<std::string>("hoDepositName");
//      produces<reco::IsoDepositMap>(hoDepositName_);
//      jetDepositName_ = iConfig.getParameter<std::string>("jetDepositName");
//      produces<reco::IsoDepositMap>(jetDepositName_);
//    }
}

/// Destructor
MuonProducer::~MuonProducer(){ 

}


/// reconstruct muons
void MuonProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup){

   const std::string metname = "Muon|RecoMuon|MuonIdentification|MuonProducer";

   // the muon collection, it will be loaded in the event
   std::auto_ptr<reco::MuonCollection> outputMuons(new reco::MuonCollection());
   // FIXME: need to update the asso map too!!!!!!
   

   edm::Handle<reco::MuonCollection> inputMuons; 
   event.getByLabel(theMuonsCollectionLabel, inputMuons);

   edm::Handle<reco::PFCandidateCollection> pfCandidates; 
   event.getByLabel(thePFCandLabel, pfCandidates);


   if(inputMuons->empty()) {
     // FIXME! Needs to check that all the default variables (the new one) are properly set
     event.put(outputMuons);
     // FIXME: need to update the asso map too!!!!!!
     return;
   }
   
   // FIXME: add the option to swith off the Muon-PF "info association".
   

   MuToPFMap muToPFMap;

   dout << "Number of PFCandidates: " << pfCandidates->size() << endl;
   foreach(const reco::PFCandidate &pfCand, *pfCandidates)
     if(abs(pfCand.pdgId()) == 13){
       muToPFMap[pfCand.muonRef()] = pfCand.p4();     
       dout << "MuonRef: " << pfCand.muonRef().id() << " " << pfCand.muonRef().key() << " PF p4: " << pfCand.p4() << endl;
     }


   dout << "Number of PFMuons: " << muToPFMap.size() << endl;
   dout << "Number of Muons in the original collection: " << inputMuons->size() << endl;


   reco::MuonRef::key_type muIndex = 0;
   foreach(const reco::Muon &inMuon, *inputMuons){
	  
     reco::MuonRef muRef(inputMuons, muIndex++);

     // Copy the muon 
     reco::Muon outMuon = inMuon;

     // search for the corresponding pf candidate
     MuToPFMap::iterator iter =  muToPFMap.find(muRef);
     if(iter != muToPFMap.end()){
       outMuon.setPFP4(iter->second);
       muToPFMap.erase(iter);
       dout << "MuonRef: " << muRef.id() << " " << muRef.key() 
	    << " Is it PF? " << outMuon.isPFMuon() 
	 // << " PF p4: " << outMuon.isPFMuon() ? outMuon.pfP4() : 0 << endl;
	    << " PF p4: " << outMuon.pfP4() << endl;
     }
     

     dout << "MuonRef: " << muRef.id() << " " << muRef.key() 
	  << " Is it PF? " << outMuon.isPFMuon() << endl;
   
     dout << "GLB "  << outMuon.isGlobalMuon()
	  << " TM "  << outMuon.isTrackerMuon()
	  << " STA " << outMuon.isStandAloneMuon() 
	  << " p4 "  << outMuon.p4() << endl;

       // FIXME: Fill iso quantities too!


       
     outputMuons->push_back(outMuon);    
   }

   dout << "Number of Muons in the new muon collection: " << outputMuons->size() << endl;
   event.put(outputMuons);
}

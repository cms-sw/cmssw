/** \class MuonProducer
 *  See header file.
 *
 *  $Date: 2011/05/31 14:47:01 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/MuonIdentification/plugins/MuonProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

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
  produces<reco::MuonTimeExtraMap>("combined");
  produces<reco::MuonTimeExtraMap>("dt");
  produces<reco::MuonTimeExtraMap>("csc");
  
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

   edm::Handle<reco::MuonCollection> inputMuons; 
   event.getByLabel(theMuonsCollectionLabel, inputMuons);
   
   edm::Handle<reco::PFCandidateCollection> pfCandidates; 
   event.getByLabel(thePFCandLabel, pfCandidates);


   // FIXME! Add map between old and new muons! (useful for the following PF step)

   // FIXME: need to update the asso map too!!!!!!
   
   // Fill timing information
   std::auto_ptr<reco::MuonTimeExtraMap> muonTimeMap(new reco::MuonTimeExtraMap());
   reco::MuonTimeExtraMap::Filler filler(*muonTimeMap);
   std::auto_ptr<reco::MuonTimeExtraMap> muonTimeMapDT(new reco::MuonTimeExtraMap());
   reco::MuonTimeExtraMap::Filler fillerDT(*muonTimeMapDT);
   std::auto_ptr<reco::MuonTimeExtraMap> muonTimeMapCSC(new reco::MuonTimeExtraMap());
   reco::MuonTimeExtraMap::Filler fillerCSC(*muonTimeMapCSC);


   edm::Handle<reco::MuonTimeExtraMap> timeMapCmb;
   edm::Handle<reco::MuonTimeExtraMap> timeMapDT;
   edm::Handle<reco::MuonTimeExtraMap> timeMapCSC;
   
   int nMuons=inputMuons->size();

   std::vector<reco::MuonTimeExtra> dtTimeColl(nMuons);
   std::vector<reco::MuonTimeExtra> cscTimeColl(nMuons);
   std::vector<reco::MuonTimeExtra> combinedTimeColl(nMuons);


   event.getByLabel(theMuonsCollectionLabel.label(),"combined",timeMapCmb);
   event.getByLabel(theMuonsCollectionLabel.label(),"dt",timeMapDT);
   event.getByLabel(theMuonsCollectionLabel.label(),"csc",timeMapCSC);
   
   

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
   unsigned int i = 0;
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

     
     // Fill timing information
     
     
     combinedTimeColl[i] = (*timeMapCmb)[muRef];
     dtTimeColl[i] = (*timeMapDT)[muRef];
     cscTimeColl[i] = (*timeMapCSC)[muRef];
          


     // FIXME: Fill iso quantities too!
    

       
     outputMuons->push_back(outMuon); 
     ++i;
   }

   dout << "Number of Muons in the new muon collection: " << outputMuons->size() << endl;
   edm::OrphanHandle<reco::MuonCollection> muonHandle = event.put(outputMuons);

   filler.insert(muonHandle, combinedTimeColl.begin(), combinedTimeColl.end());
   filler.fill();
   fillerDT.insert(muonHandle, dtTimeColl.begin(), dtTimeColl.end());
   fillerDT.fill();
   fillerCSC.insert(muonHandle, cscTimeColl.begin(), cscTimeColl.end());
   fillerCSC.fill();

   event.put(muonTimeMap,"combined");
   event.put(muonTimeMapDT,"dt");
   event.put(muonTimeMapCSC,"csc");
   
}

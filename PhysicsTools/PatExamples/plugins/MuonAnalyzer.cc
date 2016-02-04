/** \class ExampleMuonAnalyzer
 *  Analyzer of the muon objects
 *
 *  $Date: 2009/12/29 23:04:51 $
 *  $Revision: 1.7 $
 *  \author R. Bellan - CERN <riccardo.bellan@cern.ch>
 */


#include "PhysicsTools/PatExamples/plugins/MuonAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "TH1I.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;
using namespace edm;

/// Constructor
ExampleMuonAnalyzer::ExampleMuonAnalyzer(const ParameterSet& pset){

  theMuonLabel = pset.getUntrackedParameter<string>("MuonCollection");
}

/// Destructor
ExampleMuonAnalyzer::~ExampleMuonAnalyzer(){
}

void ExampleMuonAnalyzer::beginJob(){

  // Book histograms
  edm::Service<TFileService> fileService;
  hPtRec  = fileService->make<TH1F>("pT","p_{T}^{rec}",250,0,120);
  hPtReso = fileService->make<TH2F>("pT_Reso","(p_{T}^{rec}-p_{T}^{sim})/p_{T}^{sim}",250,0,120,100,-0.2,0.2);
  hNMuons = fileService->make<TH1I>("NMuons","Number of muons per event",20,0,20);
  
  hEHcal = fileService->make<TH1F>("EHCal","Energy deposit in HCAL",100,0,10);

  // ID
  hMuonType = fileService->make<TH1I>("MuonType", "Type of Muons", 5, 1, 6);
  hPtSTATKDiff = fileService->make<TH1F>("DiffPt_STA_TK","p^{TK}_{T}-p^{STA}_T",200,-50,50);
  hMuCaloCompatibility = fileService->make<TH1F>("CaloCompatibility","Muon HP using Calorimeters only",100,0,1);
  hMuSegCompatibility = fileService->make<TH1F>("SegmentCompatibility","Muon HP using segments only",100,0,1);
  hChamberMatched = fileService->make<TH1I>("NumMatchedChamber", "Number of matched chambers", 7, 0, 7);
  hMuIdAlgo = fileService->make<TH1I>("MuonIDSelectors", "Results of muon id selectors", 13, 0, 13);

  // Isolation
  hMuIso03SumPt = fileService->make<TH1F>("MuIso03SumPt","Isolation #Delta(R)=0.3: SumPt",200,0,10);
  hMuIso03CaloComb = fileService->make<TH1F>("MuIso03CaloComb","Isolation #Delta(R)=0.3: 1.2*ECAL+0.8HCAL",200,0,10);

  // 4Mu invariant mass
  h4MuInvMass = fileService->make<TH1F>("InvMass4MuSystem","Invariant mass of the 4 muons system",200,0,500);

}

void ExampleMuonAnalyzer::endJob(){
}
 

void ExampleMuonAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  
  // Get the Muon collection
  Handle<pat::MuonCollection> muons;
  event.getByLabel(theMuonLabel, muons);

  // How many muons in the event?
  hNMuons->Fill(muons->size());

  pat::MuonCollection selectedMuons;


  // Let's look inside the muon collection.
  for (pat::MuonCollection::const_iterator muon = muons->begin();  muon != muons->end(); ++muon){

    // pT spectra of muons
    hPtRec->Fill(muon->pt());

    // what is the resolution in pt? Easy! We have the association with generated information
    //    cout<<muon->pt()<<" "<<muon->genParticle()->pt()<<endl;
    if( muon->genLepton()!=0){
      double reso = (muon->pt() - muon->genLepton()->pt())/muon->genLepton()->pt();
      hPtReso->Fill(muon->genLepton()->pt(),reso);
    }

    // What is the energy deposit in HCal?
    if(muon->isEnergyValid())
      hEHcal->Fill(muon->calEnergy().had);

    // Which type of muons in the collection?
    if(muon->isStandAloneMuon())
      if(muon->isGlobalMuon())
	if(muon->isTrackerMuon()) hMuonType->Fill(1); // STA + GLB + TM
	else hMuonType->Fill(2); // STA + GLB
      else 
	if(muon->isTrackerMuon()) hMuonType->Fill(3);  // STA + TM
	else hMuonType->Fill(5); // STA
    else
      if(muon->isTrackerMuon()) hMuonType->Fill(4); // TM

    // ...mmm I want to study the relative resolution of the STA track with respect to the Tracker track.
    // or I want to look at a track stab
    if(muon->isGlobalMuon()){
      double diff =  muon->innerTrack()->pt() - muon->standAloneMuon()->pt();
      hPtSTATKDiff->Fill(diff);
    }
    
    // Muon ID quantities
    
    // Muon in CMS are usually MIP. What is the compatibility of a muon HP using only claorimeters?
    if(muon->isCaloCompatibilityValid())
      hMuCaloCompatibility->Fill(muon->caloCompatibility());

    // The muon system can also be used just as only for ID. What is the compatibility of a muon HP using only calorimeters?
    hMuSegCompatibility->Fill(muon::segmentCompatibility(*muon));


    // How many chambers have been associated to a muon track?
    hChamberMatched->Fill(muon->numberOfChambers());
    // If you look at MuonSegmentMatcher class you will see a lot of interesting quantities to look at!
    // you can get the list of matched info using matches()


    // Muon ID selection. As described in AN-2008/098  
    if(muon::isGoodMuon(*muon, muon::All))                                // dummy options - always true
      hMuIdAlgo->Fill(0);       
    if(muon::isGoodMuon(*muon, muon::AllStandAloneMuons))                 // checks isStandAloneMuon flag
      hMuIdAlgo->Fill(1);       
    if(muon::isGoodMuon(*muon, muon::AllTrackerMuons))                    // checks isTrackerMuon flag
      hMuIdAlgo->Fill(2);          
    if(muon::isGoodMuon(*muon, muon::TrackerMuonArbitrated))              // resolve ambiguity of sharing segments
      hMuIdAlgo->Fill(3);
    if(muon::isGoodMuon(*muon, muon::AllArbitrated))                      // all muons with the tracker muon arbitrated
      hMuIdAlgo->Fill(4);            
    if(muon::isGoodMuon(*muon, muon::GlobalMuonPromptTight))              // global muons with tighter fit requirements
      hMuIdAlgo->Fill(5);    
    if(muon::isGoodMuon(*muon, muon::TMLastStationLoose))                 // penetration depth loose selector
      hMuIdAlgo->Fill(6);       
    if(muon::isGoodMuon(*muon, muon::TMLastStationTight))                 // penetration depth tight selector
      hMuIdAlgo->Fill(7);       
    if(muon::isGoodMuon(*muon, muon::TM2DCompatibilityLoose))             // likelihood based loose selector
      hMuIdAlgo->Fill(8);   
    if(muon::isGoodMuon(*muon, muon::TM2DCompatibilityTight))             // likelihood based tight selector
      hMuIdAlgo->Fill(9);   
    if(muon::isGoodMuon(*muon, muon::TMOneStationLoose))                  // require one well matched segment
      hMuIdAlgo->Fill(10);        
    if(muon::isGoodMuon(*muon, muon::TMOneStationTight))                  // require one well matched segment
      hMuIdAlgo->Fill(11);        
    if(muon::isGoodMuon(*muon, muon::TMLastStationOptimizedLowPtLoose))   // combination of TMLastStation and TMOneStation
      hMuIdAlgo->Fill(12); 
    if(muon::isGoodMuon(*muon, muon::TMLastStationOptimizedLowPtTight))   // combination of TMLastStation and TMOneStation
      hMuIdAlgo->Fill(13);  



    // Isolation variables. There are many type of isolation. You can even build your own combining the output of
    // muon->isolationR03(). E.g.: 1.2*muon->isolationR03().emEt + 0.8*muon->isolationR03().hadEt
    // *** WARNING *** it is just an EXAMPLE!
    if(muon->isIsolationValid()){
      hMuIso03CaloComb->Fill(1.2*muon->isolationR03().emEt + 0.8*muon->isolationR03().hadEt);
      hMuIso03SumPt->Fill(muon->isolationR03().sumPt);
    }

    // OK, let see if we understood everything.
    // Suppose we are searching for H->ZZ->4mu. 
    // In mean the 4 muons have/are:
    // high pt (but 1 out of 4 can be at quite low pt)
    // isolated
    // so, we can make some requirements  
    if(muon->isolationR03().sumPt< 0.2){
      if(muon->isGlobalMuon() ||
	 muon::isGoodMuon(*muon, muon::TM2DCompatibilityLoose) || 
	 muon::isGoodMuon(*muon, muon::TMLastStationLoose))
	selectedMuons.push_back(*muon);
    }
  }

  /// simple selection... Do not want to write here my super-secret Higgs analysis ;-)
  if(selectedMuons.size() == 4){
    reco::Candidate::LorentzVector p4CM;
    for (pat::MuonCollection::const_iterator muon = selectedMuons.begin();  muon != selectedMuons.end(); ++muon){
      p4CM = p4CM + muon->p4();
    }
    h4MuInvMass->Fill(p4CM.mass());
  }
}
DEFINE_FWK_MODULE(ExampleMuonAnalyzer);








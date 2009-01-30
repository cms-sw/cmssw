// -*- C++ -*-
//
// Package:    JetMETHLTAnalyzer
// Class:      JetMETHLTAnalyzer
// 
/**\class JetMETHLTAnalyzer JetMETHLTAnalyzer.cc HLTriggerOffline/JetMET/src/JetMETHLTAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jochen Cammin
//         Created:  Sun Oct 14 20:47:10 CDT 2007
// $Id: JetMETHLTAnalyzer.cc,v 1.1 2008/01/28 20:25:54 cammin Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TObjString.h"
#include <TROOT.h>
#include <cmath>



#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "HLTrigger/HLTanalyzers/interface/HLTEgamma.h"
#include "HLTrigger/HLTanalyzers/interface/HLTInfo.h"
#include "HLTrigger/HLTanalyzers/interface/HLTJets.h"
//#include "HLTrigger/HLTanalyzers/interface/HLTMCtruth.h"
#include "HLTrigger/HLTanalyzers/interface/HLTMuon.h"

#include "DataFormats/Common/interface/Handle.h"

// using namespace std;
// using namespace reco;


//
// class decleration
//

class JetMETHLTAnalyzer : public edm::EDAnalyzer {
public:
  explicit JetMETHLTAnalyzer(const edm::ParameterSet&);
  ~JetMETHLTAnalyzer();
  
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void ObjectGetter(const edm::Event&, const edm::EventSetup&);  
  virtual void BookHistos();  
  // ----------member data ---------------------------


  // constants, array sizes, etc.
  const static int maxnHLT = 100;
  int nl1extjetc;
  int errCnt;
  int errMax;

  TFile *fout;

  int nHLTAccept[maxnHLT];
  int nHLTEvents;
  int nHLTriggers;
  TString HLTNames[maxnHLT];

  // Histograms -------------------------
  TH1D * histo; 
  TH1F *hAllHLT;
  TH1F *hAcceptedHLT;

  TH1F *hHLT[maxnHLT];
  TH1F *hHLTAccept[maxnHLT];

  //-------------------------------------
  // Histogram collection
  TObjArray *HLTHistArray;
  TObjArray *HLTNameArray;
  //-------------------------------------


  string str_CaloJets;
  string str_GenJets;

  std::string l1extramc_;
  
  bool _debug;

  bool _firstevent;
  bool _HasCaloJets;
  bool _HasGenJets;
  int nEvent;
  int nProgress;

  CaloJetCollection    jets;
  // HLTInfo hlt_analysis_;

  edm::Handle<CaloJetCollection> calojets;
  edm::Handle<GenJetCollection> genjets;
  edm::Handle<edm::TriggerResults> hltresults;

  edm::Handle<l1extra::L1JetParticleCollection> l1extjetc,l1extjetf,l1exttaujet;
  edm::Handle<l1extra::L1MuonParticleCollection> l1extmu;
  //  Handle<l1extra::L1JetParticleCollection> l1extjetc,l1extjetf,l1exttaujet;
//   Handle<l1extra::L1EtMissParticle> l1extmet;
// //Handle<l1extra::L1ParticleMapCollection> l1mapcoll;
//   Handle<EcalTrigPrimDigiCollection> ecal;
//   Handle<HcalTrigPrimDigiCollection> hcal;

// L1 jets
  double l1extjtcet[20];
  double l1extjtce[20];
  double l1extjtceta[20];
  double l1extjtcphi[20];

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
JetMETHLTAnalyzer::JetMETHLTAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  edm::Service<TFileService> fs;
  histo = fs->make<TH1D>("charge" , "Charges" , 200 , -2 , 2 );

  str_CaloJets       = iConfig.getParameter<string>("CaloJets");
  str_GenJets       = iConfig.getParameter<string>("GenJets");
  _debug = iConfig.getParameter<bool>("Debug"); 
  // HLTriggerResults_ = iConfig.getParameter<InputTag>( "HLTriggerResults" );
  l1extramc_ = iConfig.getParameter< std::string > ("l1extramc"); // can be l1extramctruth 
                                                                  // or l1extraParticles
  nProgress = 1000;
  nProgress = iConfig.getParameter<int>("Progress"); 
}


JetMETHLTAnalyzer::~JetMETHLTAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
JetMETHLTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
//    using namespace edm;


  if ((nEvent % nProgress)==0) printf("\n[INFO] Working on event number %i\n\n",nEvent);

      int charge = 0;
      histo->Fill( charge );


  ObjectGetter(iEvent, iSetup);

   if (!_HasCaloJets) {nEvent++; return;};

   if (_debug) cout << "====> number of CaloJets "<< calojets->size() << endl;
   if (_debug) cout << "====> number of GenJets "<< genjets->size() << endl;


   int j = 0;
   double LeadingCaloJetPt = 0.;
   for(CaloJetCollection::const_iterator jet = calojets->begin();
 	jet != calojets->end(); ++jet ) {
     if (_debug) std::cout <<" Jet " << j <<" pt = " << jet->pt() << std::endl;
     if (_debug) cout <<" Jet " << j
		      <<" pt = " << jet->pt()
		      <<" eta = " << jet->eta()
		      <<" phi = " << jet->phi() << endl;
     if (j==0) LeadingCaloJetPt = jet->pt();
     j++;
   }
   

 

   // Call the analysis modules
   //   hlt_analysis_.analyze(*hltresults);
   if (&hltresults) {
     // the following line contains some nasty object casting to get 
     // a TriggerResults object from its Handle
     int ntrigs=((const edm::TriggerResults&) *hltresults).size();
     nHLTriggers = ntrigs;
     nHLTEvents++;
     //
     if (_debug && ntrigs==0){std::cout << "%HLTInfo -- No trigger name given in TriggerResults of the input " << std::endl;}
     if (_debug && ntrigs >0){std::cout << "%HLTInfo -- Found " << ntrigs << " HLT bits" << std::endl;}


     // check that number of triggers doesn't exceed histogram bounds
     if (ntrigs > maxnHLT){
       printf("[ERROR] Too many triggers: %i. Increase histogram limit, which currently is %i.\n",ntrigs,maxnHLT);
       return;
     }


     edm::TriggerNames triggerNames_;
     triggerNames_.init((const edm::TriggerResults&) *hltresults);

    // ...Fill the corresponding accepts in branch-variables
    for (int itrig = 0; itrig != ntrigs; ++itrig){
      string trigName = triggerNames_.triggerName(itrig);
      if (_debug) printf("[INFO] Working on trigger %s\n",trigName.c_str());
      if (nHLTEvents == 1) HLTNames[itrig] = trigName.c_str();

      TObjString tos_trigName = trigName.c_str(); 
      // Add the trigger name and histogram to the collection if it's not already in there.

      int found2 = HLTNameArray->IndexOf(&tos_trigName);

      if (_debug) printf("[DEBUG]: found2 = %i\n",found2);


      if (_firstevent) {
      //      if (!HLTNameArray->FindObject(&tos_trigName)){
      //	printf("[INFO] Adding trigger %s to list\n",trigName.c_str());
	if (_debug) printf("[DEBUG] First event\n");
	//	HLTNameArray->Add(&tos_trigName);
	hHLT[itrig]->SetName(Form("hHLT_%s",trigName.c_str()));
	hHLTAccept[itrig]->SetName(Form("hHLTAccept_%s",trigName.c_str()));
	hHLT[itrig]->SetTitle(Form("hHLT_%s",trigName.c_str()));
	hHLTAccept[itrig]->SetTitle(Form("hHLTAccept_%s",trigName.c_str()));
	//	HLTHistArray->Add(hHLT[itrig]);
	//	HLTHistArray->Add(hHLTAccept[itrig]);
//       } else {
// 	printf("[INFO] Trigger %s is already in the list\n",trigName.c_str());
//       }
      }

      bool accept = ((const edm::TriggerResults&) *hltresults).accept(itrig);

      if (_debug) std::cout << "%HLTInfo --  Number of HLT Triggers: " << ntrigs << std::endl;
      if (_debug) std::cout << "%HLTInfo --  HLTTrigger(" << itrig << "): " << trigName << " = " << accept << std::endl;
      hAllHLT->Fill(itrig+0.1);
      if (accept){
	hAcceptedHLT->Fill(itrig+0.1);
	nHLTAccept[itrig]++;
      }

      //-----------------
      //      if (trigName == "CandHLT2jetAve30") {
	hHLT[itrig]->Fill(LeadingCaloJetPt);
	if (accept) hHLTAccept[itrig]->Fill(LeadingCaloJetPt);
	//      }
      //-----------------



    }
     
   } else {
     printf("[ERROR] There is no TriggerResults object!\n");
   }



   // work with l1 information
   if (&l1extjetc) {
     nl1extjetc = ((l1extra::L1JetParticleCollection&)*l1extjetc).size();
     if (_debug) printf("[Info] There are %i L1 jets.\n",nl1extjetc);
     l1extra::L1JetParticleCollection myl1jetsc;
     //     myl1jetsc= (l1extra::L1JetParticleCollection&)*l1extjetc;
// //      std::sort(myl1jetsc.begin(),myl1jetsc.end(),EtGreater());
     int il1exjt = 0;
     //     for (l1extra::L1JetParticleCollection::const_iterator jtItr = myl1jetsc.begin(); jtItr != myl1jetsc.end(); ++jtItr) {
     if (nl1extjetc > 0) {
       for (l1extra::L1JetParticleCollection::const_iterator jtItr = ((l1extra::L1JetParticleCollection&)*l1extjetc).begin(); jtItr != ((l1extra::L1JetParticleCollection&)*l1extjetc).end(); ++jtItr) {
	 l1extjtcet[il1exjt] = jtItr->et();
	 l1extjtce[il1exjt] = jtItr->energy();
	 l1extjtceta[il1exjt] = jtItr->eta();
	 l1extjtcphi[il1exjt] = jtItr->phi();

	 if (_debug){
	   printf("[INFO] L1 Jet(%i) Et = %f\n",il1exjt,l1extjtcet[il1exjt]);
	   printf("[INFO] L1 Jet(%i) E  = %f\n",il1exjt,l1extjtce[il1exjt]);
	   printf("[INFO] L1 Jet(%i) Eta= %f\n",il1exjt,l1extjtceta[il1exjt]);
	   printf("[INFO] L1 Jet(%i) Phi= %f\n",il1exjt,l1extjtcphi[il1exjt]);
	 }
	 il1exjt++;
       }
     }
   }
   else {
     nl1extjetc = 0;
     if (_debug) std::cout << "[ERROR] -- No L1 Central JET object" << std::endl;
   }






   if (_firstevent) _firstevent = false;
   nEvent++;
}


// ------------ book all histograms  ------------
void 
JetMETHLTAnalyzer::BookHistos()
{

  hAllHLT = new TH1F("hAllHLT","HLTriggers",maxnHLT,0,maxnHLT);
  hAcceptedHLT = new TH1F("hAcceptedHLT","Accepted HLTriggers",maxnHLT,0,maxnHLT);

  for (int i=0; i<maxnHLT; i++){
    hHLT[i] = new TH1F(Form("hHLT_%i",i),Form("hHLT_%i",i),500,0,500);
    hHLTAccept[i] = new TH1F(Form("hHLTAccept_%i",i),Form("hHLTAccept_%i",i),500,0,500);
  }
//   TH1F *hHLT[maxnHLT];
//   TH1F *hHLTAccept[maxnHLT];


}
// ------------ method called once each job just before starting event loop  ------------
void 
JetMETHLTAnalyzer::beginJob(const edm::EventSetup&)
{

  printf("[Starting the JetMETHLTAnalyzer]\n");

  //  maxnHLT = 100;
  errMax = 100;
  errCnt = 0;

  fout = new TFile("JetMETAnalyzer.root","RECREATE");

  // call the method to book all histograms
  BookHistos();

  HLTHistArray = new TObjArray();
  HLTNameArray = new TObjArray();

  _firstevent = true;
  nEvent = 0;
  nHLTEvents = 0;

  _HasCaloJets = true;
  _HasGenJets = true;

  nHLTriggers = 0;
  for (int i=0; i<maxnHLT; i++){
    nHLTAccept[i] = 0;
  }


}

// ------------ method called once each job just after ending the event loop  ------------
void 
JetMETHLTAnalyzer::endJob() {


  for (int i=0; i<nHLTriggers; i++){
    double frac = nHLTAccept[i]/double(nHLTEvents);
    printf("Trigger[%3i] (%35s) fired %7.2f percent\n",i,HLTNames[i].Data(),100.*frac);
    printf("Trigger[%3i] (%35s) fired %i times out of %i\n",i,HLTNames[i].Data(),nHLTAccept[i],nHLTEvents);
  }


  //  gROOT->GetList();
  HLTHistArray->Write();
  fout->Write();
  fout->Close();

  printf("[Ending the JetMETHLTAnalyzer]\n");
}

// ------------ Get object collections  ------------
void 
JetMETHLTAnalyzer::ObjectGetter(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  if (_debug) printf("[Starting JetMETHLTAnalyzer::ObjectGetter]\n");
  string errMsg("");
  // get calo jet collection
  try {iEvent.getByLabel(str_CaloJets, calojets);} catch (...) {errMsg=errMsg + " -- No CaloJets"; _HasCaloJets = false;}
  // get gen jet collection
  try {iEvent.getByLabel(str_GenJets, genjets);} catch (...) {errMsg=errMsg + " -- No GenJets"; _HasGenJets = false;}
  // get trigger results
  iEvent.getByType(hltresults);
  //iEvent.getByLabel(HLTriggerResults,hltresults);


//   // get L1 objects
  try {iEvent.getByLabel(l1extramc_,"Central",l1extjetc);} catch (...) {errMsg=errMsg + " -- No central L1Jet objects";}

  if ((errMsg != "") && (errCnt < errMax)){
    errCnt=errCnt+1;
    errMsg=errMsg + ".";
    std::cout << "%HLTAnalyzer-Warning" << errMsg << std::endl;
    if (errCnt == errMax){
      errMsg="%JetMETAnalyzer-Warning -- Maximum error count reached -- No more messages will be printed.";
      std::cout << errMsg << std::endl;    
    }
  }


}

//define this as a plug-in
DEFINE_FWK_MODULE(JetMETHLTAnalyzer);

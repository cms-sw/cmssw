// -*- C++ -*-
//
// Package:    TestTriggerStudy
// Class:      TestTriggerStudy
// 
/**\class TestTriggerStudy TestTriggerStudy.cc Test/TestTriggerStudy/src/TestTriggerStudy.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gobinda Majumder
//         Created:  Sat Nov 18 09:13:08 CET 2006
// $Id$
//
//
// 21/08/07  Add calorimeter energy/tracks along J/psi direction
// Same for muons. All are wrt its direction at vertex.
// 

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//#include "FWCore/Framework/interface/Handle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "TPostScript.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraphErrors.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include <string>

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"

#include "CLHEP/Random/Random.h"
//#include "TRandom.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace edm;
//using namespace reco;  

  static const unsigned int nL1trg = 200;

  static const unsigned int nL1mx=120;
  static const unsigned int nHLTmx=120;

  static const unsigned nHLTtr_Bphys=16;
  unsigned hltbits_Bphys[nHLTtr_Bphys+1] = {46,47,49,50,51,53,
				   55,56,57,59,65,
				   66,70,75,76,81,999} ;

  static const unsigned nHLTtr_inclB=9 ;
  unsigned hltbits_inclB[nHLTtr_inclB+1]={47,55,56,59,65,66,75,76,81,999} ;

  static const unsigned nHLTtr_Bsmumu=3 ;
  unsigned hltbits_Bsmumu[nHLTtr_Bsmumu+1]={49,57,70,999} ;

  
  unsigned po=9999999;

  unsigned l1Prescale[nL1mx] = {1000,1000,1,1,1,1, // 6 0-5
                           1,10000,1000,100,1,1, // 12 6-11
                           1,1,10000,1000,100,100, // 18 12-17
                           1,1,1,po,po,10000, // 24 18-23
                           po,100,1,1,1,po, // 30 24-29
			   po,po,po,1000,po,1, // 36 30-35
			   1,po,po,1,1,1, // 42 36-41
			   1,10000,1,1,1,1, // 48 42-47
			   1,1,1,10000,1,1, // 56 48-53
			   1,1,1000,100,po,1, // 60 54-59
			   1,1,1,20,1,1, // 66 60-65
			   1,1,1,po,20,po,  // 72 66-71 
			   1,1,1,1,po,po, // 78 72-77
			   po,po,po,po,po,po, // 84 78-83
			   po,po,po,po,po,po, // 90 84-89
			   1,1,po,1,po,po, // 96 90-95
			   1,po,po,po,po,po, // 102 96-101
			   po,po,po,po,po,po, // 108 102-107
			   po,po,po,po,po,1, // 114 108-113 
			   po,po,po,po,3000000,3000000 // 120 114-119
			  } ;

  unsigned hltPrescale[nHLTmx] =  {1,1,1,1,1,1, // 6 0-5
                             1,1,1,1,1,1, // 12 6-11
			     1,10,100,10000,100000,10000, // 18 12-17
			     3000000,3000000,100000,10000,100,10, // 24
                                                          // 18-23
                             1,1,1,1,1,1, // 30 24-29
			     1,1,1,1,1,1, // 36 30-35
			     1,1,1,1,1,1, // 42 36-41
			     1,1,1,1,1,1, // 48 42-47
			     1,1,1,1,1,1, // 54 48-53
			     1,4000,2000,400,100,1000, // 60 54-59
			     1,1,1,1,1,20, // 66 60-65
			     1,1,1,1,1,1, // 72 66-71
			     1,1,1,1,1,1, // 78 72-77
			     po,po,po,1,1,1, // 84 78-83
			     1,1,1,1,1,1     // 90 84-89
			    } ;
//
// class decleration
//

class TestTriggerStudy : public edm::EDAnalyzer {
   public:
      explicit TestTriggerStudy(const edm::ParameterSet&);
      ~TestTriggerStudy();
  

   private:
  
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  int irun, ievt, l1ndf, l2ndf, itag;
  unsigned l1trg[4], hlttr[4];
  float motherpt, mothereta, mothervtx;
  unsigned int Ntp; // # of trigger paths (should be the same for all events!)
  unsigned int Nall_trig; // # of all triggered events
  unsigned int Nevents; // # of analyzed events


  unsigned int  nErrors_;            // number of events with error (EDProduct[s] not found)
  unsigned int  nAccepts_;           // number of events accepted by (any) L1 algorithm

  bool init_;                           // vectors initialised or not

  typedef std::map<std::string, unsigned int> trigPath;

  trigPath Ntrig; // # of triggered events per path

  // # of cross-triggered events per path
  // (pairs with same name correspond to unique trigger rates for that path)
  std::map<std::string, trigPath> Ncross;

  // whether a trigger path has fired for given event
  // (variable with event-scope)
  std::map<std::string, bool> fired; 

  unsigned int l1Accepts_[nL1trg];
  std::string l1Names_[nL1trg];
  
  unsigned hltPrescaleCounter[nHLTmx] ;

  std::string theRootFileName;
  std::string theoutputpsFile;
  TFile* theFile;
  TTree* T1;

  TH1F* all_l1trg;
  TH1F* all_hlttr;

  TH1F* mother_pt;
  TH1F* mother_eta;
  TH1F* mother_vtx;

  TH1F* l1_DoubleMu3_effpT ;
  TH1F* l1_DoubleMu3_effeta ;

  TH1F* Bphys_hlt_effpT[nHLTtr_Bphys+1] ;
  TH1F* Bphys_hlt_effeta[nHLTtr_Bphys+1] ;
  TH1F* Bphys_hlt_effvtx[nHLTtr_Bphys+1] ;

  TH1F* inclB_hlt_effpT[nHLTtr_inclB+1] ;
  TH1F* inclB_hlt_effeta[nHLTtr_inclB+1] ;
  TH1F* inclB_hlt_effvtx[nHLTtr_inclB+1] ;

  TH1F* Bsmumu_hlt_effpT[nHLTtr_Bsmumu+1] ;
  TH1F* Bsmumu_hlt_effeta[nHLTtr_Bsmumu+1] ;
  TH1F* Bs_hlt_effvtx[nHLTtr_Bsmumu+1] ;

  // ----------member data ---------------------------
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

TestTriggerStudy::TestTriggerStudy(const edm::ParameterSet& pset)
{
   //now do what ever initialization is needed
  theoutputpsFile = pset.getUntrackedParameter<string>("psFileName", "test.ps");
  theRootFileName = pset.getUntrackedParameter<string>("RootFileName");
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();
  
  T1 = new TTree("T1", "Trigger"); 

  T1->Branch("irun", &irun, "irun/I");  
  T1->Branch("ievt", &ievt, "ievt/I"); 

  T1->Branch("l1trg", l1trg, "l1trg[4]/i");   
  T1->Branch("hlttr", hlttr, "hlttr[4]/i");   

  T1->Branch("motherpt", &motherpt, "motherpt/F");   
  T1->Branch("mothereta", &mothereta, "mothereta/F"); 
  T1->Branch("mothervtx", &mothervtx, "mothervtx/F");  

  all_l1trg = new TH1F("all_l1trg","all L1 trigger",nL1mx,-0.5, nL1mx-0.5);
  all_hlttr = new TH1F("all_hlttr","all HLT trigger",nHLTmx,-0.5, nHLTmx-0.5);

  
  mother_pt = new TH1F("mother_pt","p_{T} of mother particle ",50,.5, 50.5);

  mother_eta = new TH1F("mother_eta","#eta of mother particle",50,-2.0, 2.0);

  mother_vtx = new TH1F("mother_vtx","vertex position of mother particle",50, 0, 1.);

  l1_DoubleMu3_effpT = new TH1F("l1_DoubleMu3_effpT","l1 double Muon Efficiency wrt p_{T}",50, 0.5, 50.5) ;

  l1_DoubleMu3_effeta = new TH1F("l1_DoubleMu3_effeta","l1 double Muon Efficiency wrt #eta",50, -4.0, 4.0) ;

  char title[200];
  for(unsigned i=0;i<=nHLTtr_Bphys;i++){
    sprintf(title, "Bphys_hlt_effpT_%i_effpT", hltbits_Bphys[i]) ;
    Bphys_hlt_effpT[i] = new TH1F(title, title, 50, .5, 50.5) ;
    sprintf(title, "Bphys_hlt_%i_effeta", hltbits_Bphys[i]) ;
    Bphys_hlt_effeta[i] = new TH1F(title, title, 50, -4., 4.) ;

    sprintf(title, "Bphys_hlt_%i_effvtx", hltbits_Bphys[i]) ;
    Bphys_hlt_effvtx[i] = new TH1F(title, title, 50, -4., 4.) ;
  }

  for(unsigned i=0;i<=nHLTtr_inclB;i++){
    sprintf(title, "inclB_hlt_effpT_%i_effpT", hltbits_inclB[i]) ;
    inclB_hlt_effpT[i] = new TH1F(title, title, 50, .5, 50.5) ;
    sprintf(title, "inclB_hlt_%i_effeta", hltbits_inclB[i]) ;
    inclB_hlt_effeta[i] = new TH1F(title, title, 50, -4., 4.) ;
    sprintf(title, "inclB_hlt_%i_effvtx", hltbits_inclB[i]) ;
    inclB_hlt_effvtx[i] = new TH1F(title, title, 50, -4., 4.) ;
  }

  
  for (unsigned i=0;i<nHLTmx;i++) {
    hltPrescaleCounter[i]=0;
  }

  Ntp = Nall_trig = Nevents = nErrors_ = nAccepts_ = 0;
  for (unsigned i=0; i<nL1trg; i++) {l1Accepts_[i] = 0;}

}

TestTriggerStudy::~TestTriggerStudy()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  theFile->cd();
  theFile->Write();
  theFile->Close();

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TestTriggerStudy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //  using namespace edm;
  //  using namespace reco;  

  Nevents++;
  irun = iEvent.id().run();
  ievt = iEvent.id().event();

  if (Nevents%1000==1) {
    cout << endl ;
    cout << endl ;
    cout << endl ;
    cout << "********************************************************************" << endl ;
    cout << "**                                                                **" << endl ;
    cout << "**                                                                **" << endl ;
    cout << "**                                                                **" << endl ;
    cout << "TestTriggerStudy::analyze Event processing , " << Nevents <<" of Run # " << irun << " Evt # " << ievt << endl; 
    cout << "**                                                                **" << endl ;
    cout << "**                                                                **" << endl ;
    cout << "**                                                                **" << endl ;
    cout << "********************************************************************" << endl ;
  }

  for (int i=0; i<4; i++) {
    
    l1trg[i] = hlttr[i] =0;

  }

  Handle <HepMCProduct> EvtHandle;
  iEvent.getByLabel("source",EvtHandle);
  //  iEvent.getByLabel("VtxSmeared",EvtHandle);
  
  const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
  
  for ( HepMC::GenEvent::particle_const_iterator p = Evt->particles_begin(); p != Evt->particles_end(); p++ ) { 

    //    cout << "mothx = " << (*p)->pdg_id() << " " << "moth Status " << (*p)->status() << endl ;

    CLHEP::HepLorentzVector moth;

    if( //abs( (*p)->pdg_id() ) != 553 && 
       abs( (*p)->pdg_id() ) != 443 ) continue ;

    HepMC::FourVector tmp4v = (*p)->momentum() ;
    
    moth = HepLorentzVector(tmp4v.px(), tmp4v.py(), tmp4v.pz(), tmp4v.e()) ;
    
    motherpt = moth.perp();
    
    mothereta = moth.eta();

    mothervtx = 0.1*(*p)->production_vertex()->point3d().perp();

    //    cout << "motherpt = " << motherpt << " " << "mothereta = " << mothereta << " " << "mothervtx = " << mothervtx << endl ;
    
    mother_pt->Fill(motherpt);
    mother_eta->Fill(mothereta);    
    mother_vtx->Fill(mothervtx);

  }
  
  // 
  // Trigger informations
  //
  //L1 level

  const unsigned int n(l1extra::L1ParticleMap::kNumOfL1TriggerTypes);

  // get hold of L1ParticleMapCollection
  Handle<l1extra::L1ParticleMapCollection> L1PMC;
  try {iEvent.getByLabel("l1extraParticleMap",L1PMC);} catch (...) {;}
  
  if (L1PMC.isValid()) {
  } else {
    nErrors_++;
    return;
  }

  // initialisation (could be made dynamic)
  assert(n==L1PMC->size());
  if (!init_) {
    init_=true;
    //    l1Names_.resize(n);
    //    l1Accepts_.resize(n);
    for (unsigned int i=0; i!=n && i<nL1trg; ++i) {
      l1Accepts_[i]=0;
      if (i<l1extra::L1ParticleMap::kNumOfL1TriggerTypes) {
	l1extra::L1ParticleMap::L1TriggerType 
	  type(static_cast<l1extra::L1ParticleMap::L1TriggerType>(i));
	l1Names_[i]=l1extra::L1ParticleMap::triggerName(type);
      } else {
	l1Names_[i]="@@NameNotFound??";
      }

    }
  }

  // decision for each L1 algorithm
  for (unsigned int i=0; i!=n && i<nL1trg; ++i) {
    int itrg = 0;
    if ((*L1PMC)[i].triggerDecision()) { 
      all_l1trg->Fill(i);
      
      l1Accepts_[i]++; 

      //      cout << "l1 bit, i = " << i << " l1 bit name " << l1Names_[i] << endl ;

      if(i == 46){

	l1_DoubleMu3_effpT->Fill(motherpt) ;
	l1_DoubleMu3_effeta->Fill(mothereta) ;
      }

      itrg = 1;
    }
    //       if (Nevents <5) 
    //    cout <<"l1accept "<<i <<" "<<l1Names_[i]<<" "<< l1Accepts_[i]<<endl;
    if (i<128) {l1trg[int(i/32)] += pow(2., double(i%32))*itrg;}

    //    if (L1GTRR->decisionWord()[i]) l1Accepts_[i]++;
  }


  //HLT
  Handle<edm::TriggerResults> trigRes;
  //GM18/10/07  iEvent.getByLabel("TriggerResults", trigRes);
  iEvent.getByType(trigRes);

  unsigned int size = trigRes->size();
  if(Ntp) {
    assert(Ntp == size);
  } else {
    Ntp = size;
  }

  //  cout <<"Nevents = "<<Nevents<<" "<<Ntp<<" "<<(int)trigRes->accept()<<endl;
  if(trigRes->accept())++Nall_trig;
  edm::TriggerNames triggerNames(*trigRes);
  
  // loop over all paths, get trigger decision

  int itrigBphys = 0;
  int itriginclB = 0;  

  for(unsigned i = 0; i != size; ++i) {
    std::string name = triggerNames.triggerName(i);
    fired[name] = trigRes->accept(i);
    int ihlt =  trigRes->accept(i);

    if (i<128) {hlttr[int(i/32)] += pow(2., double(i%32))*ihlt;}
    
    if (ihlt!=0) { 
      all_hlttr->Fill(i);

      hltPrescaleCounter[i]++;

      for (unsigned ij=0; ij<nHLTtr_Bphys; ij++) {
	if (hltbits_Bphys[ij]==i) {
	  if (hltPrescaleCounter[i]%hltPrescale[i]==0) itrigBphys = 1;
	  Bphys_hlt_effpT[ij]->Fill(motherpt); //, 1./float(hltPrescale[i]));
	  Bphys_hlt_effeta[ij]->Fill(mothereta); //, 1./float(hltPrescale[i])); 
	  Bphys_hlt_effvtx[ij]->Fill(mothervtx); //, 1./float(hltPrescale[i])); 
	}
      }

      for (unsigned ij=0; ij<nHLTtr_inclB; ij++) {
	if (hltbits_inclB[ij]==i) {
	  if (hltPrescaleCounter[i]%hltPrescale[i]==0) itriginclB = 1;
	  inclB_hlt_effpT[ij]->Fill(motherpt); //, 1./float(hltPrescale[i]));
	  inclB_hlt_effeta[ij]->Fill(mothereta); //, 1./float(hltPrescale[i])); 
	  inclB_hlt_effvtx[ij]->Fill(mothervtx); //, 1./float(hltPrescale[i])); 
	}
      }
    }

    //    if (Nevents <5) cout <<"trigger bit "<< i <<" "<<name<<" "<<fired[name]<<endl;
    if(fired[name])
      ++(Ntrig[name]);
  }
  
  if (itrigBphys==1) {
    Bphys_hlt_effpT[nHLTtr_Bphys]->Fill(motherpt);
    Bphys_hlt_effeta[nHLTtr_Bphys]->Fill(mothereta); 
    Bphys_hlt_effvtx[nHLTtr_Bphys]->Fill(mothervtx); 
  }

  if (itriginclB==1) {
    inclB_hlt_effpT[nHLTtr_inclB]->Fill(motherpt);
    inclB_hlt_effeta[nHLTtr_inclB]->Fill(mothereta); 
    inclB_hlt_effvtx[nHLTtr_inclB]->Fill(mothervtx); 
  }

  T1->Fill();

}


// ------------ method called once each job just before starting event loop  ------------
void 
TestTriggerStudy::beginJob(const edm::EventSetup& iSetup)
{

}


//  trackdetectorassociator_.init(iSetup);


// ------------ method called once each job just after ending the event loop  ------------
void 
TestTriggerStudy::endJob() {

  theFile->cd();

  gStyle->SetOptStat(1100); 
  gStyle->SetOptFit(111); //0110);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetStatBorderSize(1);
  gStyle->SetStatStyle(1001);
  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetStatColor(10);  
  gStyle->SetStatX(0.95);
  gStyle->SetStatY(0.95);
  gStyle->SetStatW(0.16);
  gStyle->SetStatH(0.14);
  gStyle->SetTitleSize(0.045,"XYZ");
  gStyle->SetLabelSize(0.045,"XYZ");
  gStyle->SetLabelOffset(0.012,"XYZ");

  int ips=111;

  TPostScript ps(theoutputpsFile.c_str(),ips);  //FIXGM Use .cfg file for name  

  //  ps.Range(16,20);

  int xsiz = 600;
  int ysiz = 800;
  const int nbinmx = 200;
  double entries[nbinmx];
  double errorX[nbinmx];
  double errorY[nbinmx];
  double effi[nbinmx];
  double xaxis[nbinmx];
  char tGraphErrTitle[100] ;

  ps.NewPage();

  TCanvas *c0 = new TCanvas("c0", " L1 & HLT trigger", xsiz, ysiz);

  c0->Divide(1,2);

  c0->cd(1); 

  unsigned nbin = all_l1trg->GetNbinsX();
  for (unsigned i=0; i<nbin; i++) {
    entries[i] = all_l1trg->GetBinContent(i+1);
    xaxis[i] = all_l1trg->GetBinCenter(i);
    effi[i] = entries[i]/max(unsigned(1),Nevents);
    errorX[i] = 0;
    errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
  }

  TGraphErrors* gr1 = new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
  gr1->SetTitle("TGraphError_all_l1trg") ;
  gr1->SetMarkerColor(6);
  gr1->SetMarkerStyle(23);
  gr1->GetXaxis()->SetTitle("L1 trigger bits") ;
  gr1->GetXaxis()->CenterTitle();
  gr1->Draw("AP");
  
  c0->cd(2);

  nbin = all_hlttr->GetNbinsX();
  for (unsigned i=0; i<nbin; i++) {
    entries[i] = all_hlttr->GetBinContent(i+1);
    xaxis[i] = all_hlttr->GetBinCenter(i);
    effi[i] = entries[i]/max(unsigned(1),Nevents);
    errorX[i] = 0;
    errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
  }

  TGraphErrors* gr2 = new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
  gr2->SetTitle("TGraphError_all_hlttrg") ;
  gr2->SetMarkerColor(6);
  gr2->SetMarkerStyle(23);
  gr2->GetXaxis()->SetTitle("HLtrigger bits") ;
  gr2->GetXaxis()->CenterTitle();
  gr2->Draw("AP");
 
  c0->Update();

  ps.NewPage() ;

  TGraphErrors* ggr[100];

  TCanvas *c1 = new TCanvas("c1", " HLTrigger", xsiz, ysiz);

  c1->Divide(4,4) ;

  for (unsigned ij=0; ij<nHLTtr_Bphys; ij++) {
    
    c1->cd(ij+1) ;

    nbin = Bphys_hlt_effpT[ij]->GetNbinsX();
    
    for (unsigned i=0; i<nbin; i++) {
      entries[i] = Bphys_hlt_effpT[ij]->GetBinContent(i+1);
      xaxis[i] = Bphys_hlt_effpT[ij]->GetBinCenter(i);
      effi[i] = entries[i]/max(unsigned(1),Nevents);
      errorX[i] = 0;
      errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
      if (ij !=nHLTtr_Bphys) {
	effi[i] /=hltPrescale[hltbits_Bphys[ij]];
	errorY[i] /=hltPrescale[hltbits_Bphys[ij]];
      }
    }

    ggr[ij]= new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
    sprintf(tGraphErrTitle, "TGraphErr_BphysHLTbit_%i",hltbits_Bphys[ij] ) ;

    ggr[ij]->SetTitle(tGraphErrTitle) ;
    ggr[ij]->SetMarkerColor(6);
    ggr[ij]->SetMarkerStyle(23);
    ggr[ij]->GetXaxis()->SetTitle("p_{T}") ;
    ggr[ij]->GetXaxis()->CenterTitle();
    ggr[ij]->GetYaxis()->SetTitle("Efficiency") ;
    ggr[ij]->GetYaxis()->CenterTitle();
    ggr[ij]->Draw("AP");

  }

  c1->Update() ;
  
  ps.NewPage();

  TCanvas *c2 = new TCanvas("c2", " HLTrigger", xsiz, ysiz);

  c2->Divide(4,4)   ;

  for (unsigned ij=0; ij<nHLTtr_Bphys; ij++) {
    
    c2->cd(ij+1); 

    nbin = Bphys_hlt_effeta[ij]->GetNbinsX();
    
    for (unsigned i=0; i<nbin; i++) {
      entries[i] = Bphys_hlt_effeta[ij]->GetBinContent(i+1);
      xaxis[i] = Bphys_hlt_effeta[ij]->GetBinCenter(i);
      effi[i] = entries[i]/max(unsigned(1),Nevents);
      errorX[i] = 0;
      errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
      if (ij !=nHLTtr_Bphys) {
	effi[i] /=hltPrescale[hltbits_Bphys[ij]];
	errorY[i] /=hltPrescale[hltbits_Bphys[ij]];
      }
    }

    ggr[ij]= new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
    sprintf(tGraphErrTitle, "TGraphErr_BphysHLTbit_%i",hltbits_Bphys[ij] ) ;
    ggr[ij]->SetTitle(tGraphErrTitle) ;
    ggr[ij]->SetMarkerColor(6);
    ggr[ij]->SetMarkerStyle(23);
    ggr[ij]->GetXaxis()->SetTitle("#eta") ;
    ggr[ij]->GetXaxis()->CenterTitle();
    ggr[ij]->GetYaxis()->SetTitle("Efficiency") ;
    ggr[ij]->GetYaxis()->CenterTitle();
    ggr[ij]->Draw("AP");

   
  }

  c2->Update();

  ps.NewPage();

  TCanvas *c3 = new TCanvas("c3", "HLTrigger", xsiz, ysiz);

  c3->Divide(3,3) ;

  for (unsigned ij=0; ij < nHLTtr_inclB; ij++) {

    c3->cd(ij+1) ;

    nbin = inclB_hlt_effeta[ij]->GetNbinsX();    
    for (unsigned i=0; i<nbin; i++) {
      entries[i] = inclB_hlt_effeta[ij]->GetBinContent(i+1);
      xaxis[i] = inclB_hlt_effeta[ij]->GetBinCenter(i);
      effi[i] = entries[i]/max(unsigned(1),Nevents);
      errorX[i] = 0;
      errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
      if (ij !=nHLTtr_inclB) {
	effi[i] /=hltPrescale[hltbits_inclB[ij]];
	errorY[i] /=hltPrescale[hltbits_inclB[ij]];
      }

  }

    ggr[ij]= new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
    sprintf(tGraphErrTitle, "TGraphErr_inclBHLTbit_%i",hltbits_inclB[ij] ) ;
    ggr[ij]->SetTitle(tGraphErrTitle) ;
    ggr[ij]->SetMarkerColor(6);
    ggr[ij]->SetMarkerStyle(23);
    ggr[ij]->GetXaxis()->SetTitle("#eta") ;
    ggr[ij]->GetXaxis()->CenterTitle();
    ggr[ij]->GetYaxis()->SetTitle("Efficiency") ;
    ggr[ij]->GetYaxis()->CenterTitle();
    ggr[ij]->Draw("AP");

  }

  c3->Update() ;
  
  ps.NewPage();

  TCanvas *c4 = new TCanvas("c3", "HLTrigger", xsiz, ysiz);

  c4->Divide(3,3) ;

  for (unsigned ij=0; ij < nHLTtr_inclB; ij++) {

    c4->cd(ij+1) ;

    nbin = inclB_hlt_effpT[ij]->GetNbinsX();    
    for (unsigned i=0; i<nbin; i++) {
      entries[i] = inclB_hlt_effpT[ij]->GetBinContent(i+1);
      xaxis[i] = inclB_hlt_effpT[ij]->GetBinCenter(i);
      effi[i] = entries[i]/max(unsigned(1),Nevents);
      errorX[i] = 0;
      errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
      if (ij !=nHLTtr_inclB) {
	effi[i] /=hltPrescale[hltbits_inclB[ij]];
	errorY[i] /=hltPrescale[hltbits_inclB[ij]];
      }
      
    }
    
    ggr[ij]= new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
    sprintf(tGraphErrTitle, "TGraphErr_inclBHLTbit_%i",hltbits_inclB[ij] ) ;
    ggr[ij]->SetTitle(tGraphErrTitle) ;
    ggr[ij]->SetMarkerColor(6);
    ggr[ij]->SetMarkerStyle(23);
    ggr[ij]->GetXaxis()->SetTitle("p_{T}") ;
    ggr[ij]->GetXaxis()->CenterTitle();
    ggr[ij]->GetYaxis()->SetTitle("Efficiency") ;
    ggr[ij]->GetYaxis()->CenterTitle();
    ggr[ij]->Draw("AP"); 
    
  }

  c4->Update() ;

  ps.NewPage() ;

  TCanvas *c5 = new TCanvas("c5", "HLTrigger", xsiz, ysiz);

  c5->Divide(2,2) ;

  c5->cd(1) ;

  nbin = l1_DoubleMu3_effpT->GetNbinsX();
  for (unsigned i=0; i<nbin; i++) {
    entries[i] = l1_DoubleMu3_effpT->GetBinContent(i+1);
    xaxis[i] = l1_DoubleMu3_effpT->GetBinCenter(i);
    effi[i] = entries[i]/max(unsigned(1),Nevents);
    errorX[i] = 0;
    errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
  }			 

  ggr[0]= new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
  sprintf(tGraphErrTitle, "TGraphErr_l1_DoubleMu3_effpT") ;
  ggr[0]->SetTitle(tGraphErrTitle) ;
  ggr[0]->SetTitle(tGraphErrTitle) ;
  ggr[0]->SetMarkerColor(6);
  ggr[0]->SetMarkerStyle(23);
  ggr[0]->GetXaxis()->SetTitle("p_{T}") ;
  ggr[0]->GetXaxis()->CenterTitle();
  ggr[0]->GetYaxis()->SetTitle("Efficiency") ;
  ggr[0]->GetYaxis()->CenterTitle();
  ggr[0]->Draw("AP");

  c5->cd(2) ;

  nbin = l1_DoubleMu3_effeta->GetNbinsX();
  for (unsigned i=0; i<nbin; i++) {
    entries[i] = l1_DoubleMu3_effeta->GetBinContent(i+1);
    xaxis[i] = l1_DoubleMu3_effeta->GetBinCenter(i);
    effi[i] = entries[i]/max(unsigned(1),Nevents);
    errorX[i] = 0;
    errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
  }			 

  ggr[0]= new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
  sprintf(tGraphErrTitle, "TGraphErr_l1_DoubleMu3_effeta") ;
  ggr[0]->SetTitle(tGraphErrTitle) ;
  ggr[0]->SetTitle(tGraphErrTitle) ;
  ggr[0]->SetMarkerColor(6);
  ggr[0]->SetMarkerStyle(23);
  ggr[0]->GetXaxis()->SetTitle("#eta") ;
  ggr[0]->GetXaxis()->CenterTitle();
  ggr[0]->GetYaxis()->SetTitle("Efficiency") ;
  ggr[0]->GetYaxis()->CenterTitle();
  ggr[0]->Draw("AP");

  c5->cd(3) ;
    
  nbin = Bphys_hlt_effpT[3]->GetNbinsX();
  for (unsigned i=0; i<nbin; i++) {
    entries[i] = Bphys_hlt_effpT[3]->GetBinContent(i+1);
    xaxis[i] = Bphys_hlt_effpT[3]->GetBinCenter(i);
    effi[i] = entries[i]/max(unsigned(1),Nevents);
    errorX[i] = 0;
    errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
  }

   ggr[0]= new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
   sprintf(tGraphErrTitle, "TGraphErr_promptJ/#psiProduction") ;
   ggr[0]->SetTitle(tGraphErrTitle) ;
   ggr[0]->SetMarkerColor(6);
   ggr[0]->SetMarkerStyle(23);
   ggr[0]->GetXaxis()->SetTitle("p_{T}") ;
   ggr[0]->GetXaxis()->CenterTitle();
   ggr[0]->GetYaxis()->SetTitle("Efficiency") ;
   ggr[0]->GetYaxis()->CenterTitle();
   ggr[0]->Draw("AP");

  c5->cd(4) ;
    
  nbin = Bphys_hlt_effeta[3]->GetNbinsX();
  for (unsigned i=0; i<nbin; i++) {
    entries[i] = Bphys_hlt_effeta[3]->GetBinContent(i+1);
    xaxis[i] = Bphys_hlt_effeta[3]->GetBinCenter(i);
    effi[i] = entries[i]/max(unsigned(1),Nevents);
    errorX[i] = 0;
    errorY[i] = sqrt(effi[i]*(1-effi[i])/max(unsigned(1),Nevents));
  }

   ggr[0]= new TGraphErrors(nbin,xaxis,effi,errorX,errorY);
   sprintf(tGraphErrTitle, "TGraphErr_promptJ/#psiProduction") ;
   ggr[0]->SetTitle(tGraphErrTitle) ;
   ggr[0]->SetMarkerColor(6);
   ggr[0]->SetMarkerStyle(23);
   ggr[0]->GetXaxis()->SetTitle("#eta") ;
   ggr[0]->GetXaxis()->CenterTitle();
   ggr[0]->GetYaxis()->SetTitle("Efficiency") ;
   ggr[0]->GetYaxis()->CenterTitle();
   ggr[0]->Draw("AP");

   c5->Update() ;

   ps.Close();
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestTriggerStudy);
// /localdata/gobinda/mc/digi/HLTPoolOutput_psi2stojpsi2_145_sim_27_1.root

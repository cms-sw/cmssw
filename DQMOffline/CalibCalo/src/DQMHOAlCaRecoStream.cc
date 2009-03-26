// -*- C++ -*-
//
// Package:    DQMHOAlCaRecoStream
// Class:      DQMHOAlCaRecoStream
// 
/**\class DQMHOAlCaRecoStream DQMHOAlCaRecoStream.cc DQMOffline/DQMHOAlCaRecoStream/src/DQMHOAlCaRecoStream.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gobinda Majumder
//         Created:  Mon Mar  2 12:33:08 CET 2009
// $Id: DQMHOAlCaRecoStream.cc,v 1.1 2009/03/16 16:03:32 kodolova Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TStyle.h"

#include <string>

#include <iostream>
#include <fstream>
#include <iomanip>

//
// class decleration
//
using namespace std;
using namespace edm;

class DQMHOAlCaRecoStream : public edm::EDAnalyzer {
   public:
      explicit DQMHOAlCaRecoStream(const edm::ParameterSet&);
      ~DQMHOAlCaRecoStream();

   private:

      DQMStore* dbe_; 

      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


  MonitorElement* hMuonMultipl;
  MonitorElement* hMuonMom;
  MonitorElement* hMuonEta;
  MonitorElement* hMuonPhi;

  MonitorElement* hDirCosine;
  MonitorElement* hHOTime;
  
  MonitorElement* hSigRing[5];
  //  MonitorElement* hSigRingm1;
  //  MonitorElement* hSigRing00;
  //  MonitorElement* hSigRingp1;
  //  MonitorElement* hSigRingp2;

  MonitorElement* hPedRing[5];
  //  MonitorElement* hPedRingm1;
  //  MonitorElement* hPedRing00;
  //  MonitorElement* hPedRingp1;
  //  MonitorElement* hPedRingp2;

  MonitorElement* hSignal3x3[9];

  int Nevents;
  int Nmuons;

  std::string theRootFileName;
  std::string folderName_;
  double m_sigmaValue;

  double m_lowRadPosInMuch;
  double m_highRadPosInMuch;

  int    m_nbins;
  double m_lowEdge;
  double m_highEdge;

  bool saveToFile_;

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
DQMHOAlCaRecoStream::DQMHOAlCaRecoStream(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  
  theRootFileName = iConfig.getUntrackedParameter<string>("RootFileName","tmp.root");
  folderName_ = iConfig.getUntrackedParameter<string>("folderName");
  m_sigmaValue = iConfig.getUntrackedParameter<double>("sigmaval",0.2);
  m_lowRadPosInMuch = iConfig.getUntrackedParameter<double>("lowradposinmuch",400.0);  
  m_highRadPosInMuch = iConfig.getUntrackedParameter<double>("highradposinmuch",480.0);  
  m_lowEdge = iConfig.getUntrackedParameter<double>("lowedge",-2.0);  
  m_highEdge = iConfig.getUntrackedParameter<double>("highedge",6.0);  
  m_nbins = iConfig.getUntrackedParameter<int>("nbins",40);
  saveToFile_ = iConfig.getUntrackedParameter<bool>("saveToFile",false);
}


DQMHOAlCaRecoStream::~DQMHOAlCaRecoStream()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DQMHOAlCaRecoStream::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  Nevents++;
  
  edm::Handle<HOCalibVariableCollection>HOCalib;
  bool isCosMu = true;
  try {
    iEvent.getByType(HOCalib); 
  } catch ( cms::Exception &iEvent ) { isCosMu = false; } 
  
  if (isCosMu) { 
    hMuonMultipl->Fill((*HOCalib).size(),1.);
    if ((*HOCalib).size() >0 ) {
      for (HOCalibVariableCollection::const_iterator hoC=(*HOCalib).begin(); hoC!=(*HOCalib).end(); hoC++){
// OK!!!!	
	float okt = 2.;
	double okx = pow((*hoC).trkvx,okt) + pow((*hoC).trkvy,okt);
///////	
	double dr=pow( okx, 0.5);
	if (dr <m_lowRadPosInMuch || dr >m_highRadPosInMuch) continue; 
	
	if ((*hoC).isect <0) continue;
	if (fabs((*hoC).trkth-acos(-1.)/2)<0.000001) continue;
	int ieta = int((abs((*hoC).isect)%10000)/100.)-30;
	
	if (abs(ieta)>=16) continue;
	
	Nmuons++;
	
	
	hMuonMom->Fill((*hoC).trkmm, 1.0);
	hMuonEta->Fill(-log(tan((*hoC).trkth/2.0)), 1.0);
	hMuonPhi->Fill((*hoC).trkph, 1.0);
	hDirCosine->Fill((*hoC).hoang, 1.0);
	hHOTime->Fill((*hoC).htime, 1.0);
	
	double energy = (*hoC).hosig[4];
	double pedval = (*hoC).hocro;
	int iring = 0;
	if (ieta >=-15 && ieta <=-11) {iring = -2;}
	else if (ieta >=-10 && ieta <=-5)  {iring = -1;}
	else if (ieta >=  5 && ieta <= 10) {iring = 1;}
	else if (ieta >= 11 && ieta <= 15) {iring = 2;}
	
	hSigRing[iring+2]->Fill(energy,1.0);
	hPedRing[iring+2]->Fill(pedval,1.0);
	
	for (int k=0; k<9; k++) {
	  hSignal3x3[k]->Fill((*hoC).hosig[k]);
	}	
      } //for (HOCalibVariableCollection::const_iterator hoC=(*HOCalib).begin()
    } // if ((*HOCalib).size() >0 ) { 
  } // if (isCosMu) { 
}


// ------------ method called once each job just before starting event loop  ------------
void 
DQMHOAlCaRecoStream::beginJob(const edm::EventSetup&)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder(folderName_);

  char title[200];
  char name[200];

  hMuonMom = dbe_->book1D("hMuonMom", "Muon momentum (GeV)", 50, -100, 100);
  hMuonMom ->setAxisTitle("Muon momentum (GeV)",1);
  
  hMuonEta = dbe_->book1D("hMuonEta", "Pseudo-rapidity of muon", 50, -1.5, 1.5);
  hMuonEta ->setAxisTitle("Pseudo-rapidity of muon",1);
  
  hMuonPhi = dbe_->book1D("hMuonPhi", "Azimuthal angle of muon", 24, -acos(-1), acos(-1));
  hMuonPhi ->setAxisTitle("Azimuthal angle of muon",1);
    
  hMuonMultipl = dbe_->book1D("hMuonMultipl", "Muon Multiplicity", 10, 0.5, 10.5); 
  hMuonMultipl ->setAxisTitle("Muon Multiplicity",1);

  hDirCosine = dbe_->book1D("hDirCosine", "Direction Cosine of muon at HO tower", 50, -1., 1.);
  hDirCosine ->setAxisTitle("Direction Cosine of muon at HO tower",1);

  hHOTime = dbe_->book1D("hHOTime", "HO time distribution", 60, -20, 100.);  
  hHOTime ->setAxisTitle("HO time distribution", 1);

  for (int i=0; i<5; i++) {
    sprintf(name, "hSigRing_%i", i-2);
    sprintf(title, "HO signal in Ring_%i", i-2);
    hSigRing[i] = dbe_->book1D(name, title, m_nbins, m_lowEdge, m_highEdge);
    hSigRing[i]->setAxisTitle(title,1);

    sprintf(name, "hPedRing_%i", i-2);
    sprintf(title, "HO Pedestal in Ring_%i", i-2);
    hPedRing[i] = dbe_->book1D(name, title, m_nbins, m_lowEdge, m_highEdge);
    hPedRing[i]->setAxisTitle(title,1);
  }

  //  hSigRingm1 = dbe_->book1D("hSigRingm1", "HO signal in Ring-1", m_nbins, m_lowEdge, m_highEdge);
  //  hSigRingm1->setAxisTitle("HO signal in Ring-1",1);

  //  hSigRing00 = dbe_->book1D("hSigRing00", "HO signal in Ring_0", m_nbins, m_lowEdge, m_highEdge);
  //  hSigRing00->setAxisTitle("HO signal in Ring_0",1);

  //  hSigRingp1 = dbe_->book1D("hSigRingp1", "HO signal in Ring-1", m_nbins, m_lowEdge, m_highEdge);
  //  hSigRingp1->setAxisTitle("HO signal in Ring+1",1);

  //  hSigRingp2 = dbe_->book1D("hSigRingp2", "HO signal in Ring-2", m_nbins, m_lowEdge, m_highEdge);
  //  hSigRingp2->setAxisTitle("HO signal in Ring+2",1);
  
  //  hPedRingm2 = dbe_->book1D("hPedRingm2", "HO pedestal in Ring-2", m_nbins, m_lowEdge, m_highEdge);
  //  hPedRingm1 = dbe_->book1D("hPedRingm1", "HO pedestal in Ring-1", m_nbins, m_lowEdge, m_highEdge);
  //  hPedRing00 = dbe_->book1D("hPedRing00", "HO pedestal in Ring_0", m_nbins, m_lowEdge, m_highEdge);
  //  hPedRingp1 = dbe_->book1D("hPedRingp1", "HO pedestal in Ring-1", m_nbins, m_lowEdge, m_highEdge);
  //  hPedRingp2 = dbe_->book1D("hPedRingp2", "HO pedestal in Ring-2", m_nbins, m_lowEdge, m_highEdge);

  for (int i=-1; i<=1; i++) {
    for (int j=-1; j<=1; j++) {
      int k = 3*(i+1)+j+1;
      
      sprintf(title, "hSignal3x3_deta%i_dphi%i", i, j);
      hSignal3x3[k] = dbe_->book1D(title, title, m_nbins, m_lowEdge, m_highEdge);
      hSignal3x3[k]->setAxisTitle(title,1);
    }
  }

  Nevents = 0;
  Nmuons = 0;

}

// ------------ method called once each job just after ending the event loop  ------------
void 
DQMHOAlCaRecoStream::endJob() {
  if (dbe_) {

    //    char name[100];
    
    //    double scale = 1./max(1,Nevents);
    double scale = 1./max(1,Nmuons);
    hMuonMom->getTH1F()->Scale(scale);
    hMuonEta->getTH1F()->Scale(scale);
    hMuonPhi->getTH1F()->Scale(scale);
    hDirCosine->getTH1F()->Scale(scale);
    hHOTime->getTH1F()->Scale(scale);
    
    //    scale = 1./max(1,Nmuons);
    for (int k=0; k<5; k++) {
      hSigRing[k]->getTH1F()->Scale(scale);
      hPedRing[k]->getTH1F()->Scale(scale);
    }
    
    for (int k=0; k<9; k++) {
      hSignal3x3[k]->getTH1F()->Scale(scale);
    }

    /*    
    gStyle->SetOptLogy(0);
    gStyle->SetTitleFillColor(10);
    gStyle->SetStatColor(10);
    
    gStyle->SetCanvasColor(10);
    gStyle->SetOptStat(0); //1110);
    gStyle->SetOptTitle(1);
    
    gStyle->SetTitleColor(10);
    gStyle->SetTitleFontSize(0.09);
    gStyle->SetTitleOffset(-0.05);
    gStyle->SetTitleBorderSize(1);
    
    gStyle->SetPadColor(10);
    gStyle->SetPadBorderMode(0);
    gStyle->SetStatColor(10);
    gStyle->SetPadBorderMode(0);
    gStyle->SetStatBorderSize(1);
    gStyle->SetStatFontSize(.07);
    
    gStyle->SetStatStyle(1001);
    gStyle->SetOptFit(101);
    gStyle->SetCanvasColor(10);
    gStyle->SetCanvasBorderMode(0);
    
    gStyle->SetStatX(.99);
    gStyle->SetStatY(.99);
    gStyle->SetStatW(.45);
    gStyle->SetStatH(.16);
    gStyle->SetLabelSize(0.075,"XY");  
    gStyle->SetLabelOffset(0.21,"XYZ");
    gStyle->SetTitleSize(0.065,"XY");  
    gStyle->SetTitleOffset(0.06,"XYZ");
    gStyle->SetPadTopMargin(.09);
    gStyle->SetPadBottomMargin(0.11);
    gStyle->SetPadLeftMargin(0.12);
    gStyle->SetPadRightMargin(0.15);
    gStyle->SetPadGridX(3);
    gStyle->SetPadGridY(3);
    gStyle->SetGridStyle(2);
    gStyle->SetNdivisions(303,"XY");
    
    gStyle->SetMarkerSize(0.60);
    gStyle->SetMarkerColor(2);
    gStyle->SetMarkerStyle(20);
    gStyle->SetTitleFontSize(0.07);
    
    gStyle->SetOptLogy(0);
    TCanvas *c0x = new TCanvas("c0x", "Muon kinematics", 800, 400);
    c0x->Divide(3,1);
    c0x->cd(1); hMuonMom->getTH1F()->Draw();
    c0x->cd(2); hMuonEta->getTH1F()->Draw();
    c0x->cd(3); hMuonPhi->getTH1F()->Draw();
    
    sprintf(name, "DQM_muon_kine.jpg");
    c0x->SaveAs(name);
    delete c0x;
    
    TCanvas *c0y = new TCanvas("c0y", "HO and muon ", 800, 400);
    c0y->Divide(2,1);
    c0y->cd(1); hDirCosine->getTH1F()->Draw();
    c0y->cd(2); hHOTime->getTH1F()->Draw();
    
    sprintf(name, "DQM_hotime_ang.jpg");
    //      sprintf(out_file, "DQM_hosig_ring_%i.jpg",irunold); 
    c0x->SaveAs(name);
    delete c0y;    

    gStyle->SetOptLogy(1);
    gStyle->SetPadTopMargin(.1);
    gStyle->SetPadLeftMargin(.15);
    //    xsiz = 800;
    //    ysiz = 500;
    TCanvas *c1x = new TCanvas("c1x", "Signal in each ring", 800, 600);
    
    c1x->Divide(3,2);
    for (int ij=0; ij<5; ij++) {
      
      hSigRing[ij]->getTH1F()->GetXaxis()->SetTitleSize(0.060);
      hSigRing[ij]->getTH1F()->GetXaxis()->SetTitleOffset(1.05);
      hSigRing[ij]->getTH1F()->GetXaxis()->CenterTitle();
      hSigRing[ij]->getTH1F()->GetXaxis()->SetLabelSize(0.065);
      hSigRing[ij]->getTH1F()->GetXaxis()->SetLabelOffset(0.01);
      
      hSigRing[ij]->getTH1F()->GetYaxis()->SetLabelSize(0.065);
      hSigRing[ij]->getTH1F()->GetYaxis()->SetLabelOffset(0.01); 
      
      hSigRing[ij]->getTH1F()->SetLineWidth(3);  
      hSigRing[ij]->getTH1F()->SetLineColor(4);  
      
      c1x->cd(ij+1); hSigRing[ij]->getTH1F()->Draw();
      
      hPedRing[ij]->getTH1F()->SetLineWidth(2);  
      hPedRing[ij]->getTH1F()->SetLineColor(2);
      hPedRing[ij]->getTH1F()->Draw("same");
    }
    
    sprintf(name, "DQM_hosig_ring.jpg");
    //      sprintf(out_file, "DQM_hosig_ring_%i.jpg",irunold); 
    c1x->SaveAs(name);
    delete c1x;
    
    TCanvas *c1y = new TCanvas("c1y", "Signal in each ring", 800, 600);
    
    c1y->Divide(3,3);
    for (int ij=0; ij<9; ij++) {
      
      hSignal3x3[ij]->getTH1F()->GetXaxis()->SetTitleSize(0.060);
      hSignal3x3[ij]->getTH1F()->GetXaxis()->SetTitleOffset(1.05);
      hSignal3x3[ij]->getTH1F()->GetXaxis()->CenterTitle();
      hSignal3x3[ij]->getTH1F()->GetXaxis()->SetLabelSize(0.065);
      hSignal3x3[ij]->getTH1F()->GetXaxis()->SetLabelOffset(0.01);
      
      hSignal3x3[ij]->getTH1F()->GetYaxis()->SetLabelSize(0.065);
      hSignal3x3[ij]->getTH1F()->GetYaxis()->SetLabelOffset(0.01); 
      
      hSignal3x3[ij]->getTH1F()->SetLineWidth(3);  
      hSignal3x3[ij]->getTH1F()->SetLineColor(4);  
      
      c1y->cd(ij+1); hSignal3x3[ij]->getTH1F()->Draw();
      
    }
    
    sprintf(name, "DQM_ho_projection.jpg");
    c1y->SaveAs(name);
    delete c1y;
    */

     if (saveToFile_) dbe_->save(theRootFileName); 
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(DQMHOAlCaRecoStream);

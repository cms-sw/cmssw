// -*- C++ -*-
//
// Package:    testEcalGetWindow
// Class:      testEcalGetWindow
// 
/**\class testEcalGetWindow testEcalGetWindow.cc test/testEcalGetWindow/src/testEcalGetWindow.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include <TCanvas.h>
#include <TVirtualPad.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TH2F.h>
#include <TBox.h>

#include <iostream>

//
// class decleration
//

class testEcalGetWindow : public edm::one::EDAnalyzer<> {
public:
  explicit testEcalGetWindow( const edm::ParameterSet& );
  ~testEcalGetWindow();
  
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  // ----------member data ---------------------------
  void build(const CaloGeometry& cg, const CaloTopology& etmap, DetId::Detector det, int subdetn, const char* name);
  int towerColor(const EcalTrigTowerDetId& theTower);
  int pass_;
};

testEcalGetWindow::testEcalGetWindow( const edm::ParameterSet& /*iConfig*/ )
{
   //now do what ever initialization is needed
  pass_=0;
  // some setup for root
  gROOT->SetStyle("Plain");          // white fill colors etc.
  gStyle->SetPaperSize(TStyle::kA4);
}


testEcalGetWindow::~testEcalGetWindow()
{}

void testEcalGetWindow::build(const CaloGeometry& /*cg*/, const CaloTopology& ct, DetId::Detector det, int subdetn, const char* name) 
{
  if (det == DetId::Ecal && subdetn == EcalEndcap) 
    {
      TCanvas *canv = new TCanvas("c","",1000,1000);
      canv->SetLeftMargin(0.15);
      canv->SetBottomMargin(0.15);
      
      gStyle->SetOptStat(0);
      TH2F *h = new TH2F("","",
			 10,0.5, 100.5,
			 10,0.5, 100.5);
      
      h->Draw();
      //gPad->SetGridx();
      //gPad->SetGridy();
      gPad->Update();
      
      h->SetXTitle("x index");
      h->SetYTitle("y index");
      
      h->GetXaxis()->SetTickLength(-0.03);
      h->GetYaxis()->SetTickLength(-0.03);
      
      h->GetXaxis()->SetLabelOffset(0.03);
      h->GetYaxis()->SetLabelOffset(0.03);
      
      h->GetXaxis()->SetLabelSize(0.04);
      h->GetYaxis()->SetLabelSize(0.04);
      
      // axis titles
      h->GetXaxis()->SetTitleSize(0.04);
      h->GetYaxis()->SetTitleSize(0.04);
      
      h->GetXaxis()->SetTitleOffset(1.8);
      h->GetYaxis()->SetTitleOffset(1.9);
      
      h->GetXaxis()->CenterTitle(1);
      h->GetYaxis()->CenterTitle(1);  
      const CaloSubdetectorTopology* topology=ct.getSubdetectorTopology(det,subdetn);

      std::vector<DetId> eeDetIds;
      eeDetIds.push_back(EEDetId(1,50,1,EEDetId::XYMODE));
      eeDetIds.push_back(EEDetId(25,50,1,EEDetId::XYMODE));
      eeDetIds.push_back(EEDetId(50,1,1,EEDetId::XYMODE));
      eeDetIds.push_back(EEDetId(50,25,1,EEDetId::XYMODE));
      eeDetIds.push_back(EEDetId(3,60,1,EEDetId::XYMODE));
      for (unsigned int i=0;i<eeDetIds.size();i++)
	{

	  EEDetId myId(eeDetIds[i]);
	  if (myId.zside()==-1)
	    continue;
	  std::vector<DetId> myNeighbours=topology->getWindow(myId,13,13);
	  for (unsigned int i=0;i<myNeighbours.size();i++)
	    {
	      EEDetId myEEId(myNeighbours[i]);
	      TBox *box = new TBox(myEEId.ix()-0.5,myEEId.iy()-0.5,myEEId.ix()+0.5,myEEId.iy()+0.5);
	      box->SetFillColor(1);
	      box->Draw();
	    }

	}
      gPad->SaveAs(name);
      delete canv;
      delete h;
    }

  if (det == DetId::Ecal && subdetn == EcalBarrel) 
    {
      TCanvas *canv = new TCanvas("c","",1000,1000);
      canv->SetLeftMargin(0.15);
      canv->SetBottomMargin(0.15);
      
      gStyle->SetOptStat(0);
      TH2F *h = new TH2F("","",
			 10,-85.5, 85.5,
			 10,0.5, 360.5);
      
      h->Draw();
      //gPad->SetGridx();
      //gPad->SetGridy();
      gPad->Update();
      
      h->SetXTitle("eta index");
      h->SetYTitle("phi index");
      
      h->GetXaxis()->SetTickLength(-0.03);
      h->GetYaxis()->SetTickLength(-0.03);
      
      h->GetXaxis()->SetLabelOffset(0.03);
      h->GetYaxis()->SetLabelOffset(0.03);
      
      h->GetXaxis()->SetLabelSize(0.04);
      h->GetYaxis()->SetLabelSize(0.04);
      
      // axis titles
      h->GetXaxis()->SetTitleSize(0.04);
      h->GetYaxis()->SetTitleSize(0.04);
      
      h->GetXaxis()->SetTitleOffset(1.8);
      h->GetYaxis()->SetTitleOffset(1.9);
      
      h->GetXaxis()->CenterTitle(1);
      h->GetYaxis()->CenterTitle(1);  
      const CaloSubdetectorTopology* topology=ct.getSubdetectorTopology(det,subdetn);
      std::vector<DetId> ebDetIds;
      ebDetIds.push_back(EBDetId(1,1));
      ebDetIds.push_back(EBDetId(30,30));
      ebDetIds.push_back(EBDetId(-1,120));
      ebDetIds.push_back(EBDetId(85,1));
      for (unsigned int i=0;i<ebDetIds.size();i++)
	{
	  EBDetId myId(ebDetIds[i]);
	  std::vector<DetId> myNeighbours=topology->getWindow(myId,13,13);
	  for (unsigned int i=0;i<myNeighbours.size();i++)
	    {
	      EBDetId myEBId(myNeighbours[i]);
	      TBox *box = new TBox(myEBId.ieta()-0.5,myEBId.iphi()-0.5,myEBId.ieta()+0.5,myEBId.iphi()+0.5);
	      box->SetFillColor(1);
	      box->Draw();
	    }
	}
      gPad->SaveAs(name);
      delete canv;
      delete h;
    }
}
// ------------ method called to produce the data  ------------
void
testEcalGetWindow::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
   
   std::cout << "Here I am " << std::endl;

   edm::ESHandle<CaloTopology> theCaloTopology;
   iSetup.get<CaloTopologyRecord>().get(theCaloTopology);     

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);     

   if (pass_==1) {
     build(*pG,*theCaloTopology,DetId::Ecal,EcalBarrel,"EBGetWindowTest.eps");
   }
   if (pass_==2) {
     build(*pG,*theCaloTopology,DetId::Ecal,EcalEndcap,"EEGetWindowTest.eps");
   }
   
   pass_++;
      
}

//define this as a plug-in

DEFINE_FWK_MODULE(testEcalGetWindow);

// -*- C++ -*-
//
// Package:    dumpEcalTrigTowerMapping
// Class:      dumpEcalTrigTowerMapping
// 
/**\class dumpEcalTrigTowerMapping dumpEcalTrigTowerMapping.cc test/dumpEcalTrigTowerMapping/src/dumpEcalTrigTowerMapping.cc

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
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

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

class dumpEcalTrigTowerMapping : public edm::one::EDAnalyzer<> {
public:
  explicit dumpEcalTrigTowerMapping( const edm::ParameterSet& );
  ~dumpEcalTrigTowerMapping();
    
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  // ----------member data ---------------------------
  void build(const CaloGeometry& cg, const EcalTrigTowerConstituentsMap& etmap, DetId::Detector det, int subdetn, const char* name);
  int towerColor(const EcalTrigTowerDetId& theTower);
  int pass_;
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
dumpEcalTrigTowerMapping::dumpEcalTrigTowerMapping( const edm::ParameterSet& /*iConfig*/ )
{
   //now do what ever initialization is needed
  pass_=0;
  // some setup for root
  gROOT->SetStyle("Plain");          // white fill colors etc.
  gStyle->SetPaperSize(TStyle::kA4);
}


dumpEcalTrigTowerMapping::~dumpEcalTrigTowerMapping()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

int dumpEcalTrigTowerMapping::towerColor(const EcalTrigTowerDetId& theTower)
{
  int iEtaColorIndex=(theTower.ietaAbs()-1)%2;
  int iPhiColorIndex = 0;
  if (theTower.ietaAbs() < 26 )
    iPhiColorIndex=(theTower.iphi()-1)%2;
  else
    iPhiColorIndex=((theTower.iphi()-1)%4)/2;

  return iEtaColorIndex*2+iPhiColorIndex+1;
}
//
// member functions
//
void dumpEcalTrigTowerMapping::build(const CaloGeometry& cg, const EcalTrigTowerConstituentsMap& etmap, DetId::Detector det, int subdetn, const char* name) 
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
      const std::vector<DetId>& eeDetIds= cg.getValidDetIds(det,subdetn);

      std::cout<<"*** testing endcap trig tower mapping **"<<std::endl;
      for (unsigned int i=0;i<eeDetIds.size();i++)
	{
	  EEDetId myId(eeDetIds[i]);
	  EcalTrigTowerDetId myTower=etmap.towerOf(eeDetIds[i]);      

//	  std::cout<<"eedetid="<<EEDetId(eeDetIds[i])<<", myTower="<<myTower<<std::endl;

	  assert( myTower == EcalTrigTowerDetId::detIdFromDenseIndex( myTower.denseIndex() ) ) ;

	  if (myId.zside()==1)
	    continue;

	  TBox *box = new TBox(myId.ix()-0.5,myId.iy()-0.5,myId.ix()+0.5,myId.iy()+0.5);
	  box->SetFillColor(towerColor(myTower));
	  box->Draw();
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
      const std::vector<DetId>& ebDetIds= cg.getValidDetIds(det,subdetn);

      std::cout<<"*** testing barrel trig tower mapping **"<<std::endl;
      for (unsigned int i=0;i<ebDetIds.size();i++)
	{
	  EBDetId myId(ebDetIds[i]);
	  EcalTrigTowerDetId myTower=etmap.towerOf(ebDetIds[i]);      

	  assert( myTower == EcalTrigTowerDetId::detIdFromDenseIndex( myTower.denseIndex() ) ) ;

	  TBox *box = new TBox(myId.ieta()-0.5,myId.iphi()-0.5,myId.ieta()+0.5,myId.iphi()+0.5);
	  box->SetFillColor(towerColor(myTower));
	  box->Draw();
	}
      gPad->SaveAs(name);
      delete canv;
      delete h;
    }
}
// ------------ method called to produce the data  ------------
void
dumpEcalTrigTowerMapping::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
   
   std::cout << "Here I am " << std::endl;

   edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap;
   iSetup.get<IdealGeometryRecord>().get(eTTmap);     

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);     

   if (pass_==1) {
     build(*pG,*eTTmap,DetId::Ecal,EcalBarrel,"EBTTmapping.eps");
   }
   if (pass_==2) {
     build(*pG,*eTTmap,DetId::Ecal,EcalEndcap,"EETTmapping.eps");
   }
   
   pass_++;
      
}

//define this as a plug-in

DEFINE_FWK_MODULE(dumpEcalTrigTowerMapping);

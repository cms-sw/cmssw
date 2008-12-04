/*
 *  Hcal Pedestal Validator
 *  Steven Won, Northwestern University
 *  Based on code written by Andy Kubik, Northwestern University
 */

#include "CalibCalorimetry/HcalStandardModules/interface/HcalPedestalsValidation.h"
using namespace edm;

HcalPedestalsValidation::HcalPedestalsValidation(const ParameterSet& pset){
  firsttime = true;
  // Book histograms here
  hbenergy = new TH1F("hbenergy","HB",210,-3.,3.);
  heenergy = new TH1F("heenergy","HE",210,-3.,3.);
  hoenergy = new TH1F("hoenergy","HO",210,-3.,3.);
  hfenergy = new TH1F("hfenergy","HF",210,-3.,3.);
}

HcalPedestalsValidation::~HcalPedestalsValidation(){

   double hbe = hbenergy->GetMean();
   double hbw = hbenergy->GetRMS();
   double hee = heenergy->GetMean();
   double hew = heenergy->GetRMS();
   double hoe = hoenergy->GetMean();
   double how = hoenergy->GetRMS();
   double hfe = hfenergy->GetMean();
   double hfw = hfenergy->GetRMS();

   if( (fabs(hbe) > .001) || (fabs(hee) > .001) || (fabs(hoe) > .001) || (fabs(hfe) > .001) ) 
     std::ofstream Touch("energyflag.bool");

   std::ofstream energyout("energies.txt");
   energyout << "HB: " << hbe << "  " << hbw << std::endl;
   energyout << "HE: " << hee << "  " << hew << std::endl;
   energyout << "HO: " << hoe << "  " << hfw << std::endl;
   energyout << "HF: " << hfe << "  " << how << std::endl;
   energyout.close();

   gStyle->SetCanvasDefW(1200);
   gStyle->SetCanvasDefH(800);

   TCanvas * c1 = new TCanvas("c1","graph",1);
   c1->Divide(2,2);
   c1->cd(1);
   hbenergy->Draw();
   c1->cd(2);
   heenergy->Draw();
   c1->cd(3);
   hoenergy->Draw();
   c1->cd(4);
   hfenergy->Draw();
   c1->SaveAs(outFileName.c_str());
}

void HcalPedestalsValidation::analyze(const Event & event, const EventSetup& eventSetup){

   edm::Handle<HBHERecHitCollection> hbherh;  event.getByLabel("hbhereco",hbherh);
   edm::Handle<HORecHitCollection> horh;  event.getByLabel("horeco",horh);
   edm::Handle<HFRecHitCollection> hfrh;  event.getByLabel("hfreco",hfrh);

   HBHERecHitCollection::const_iterator hbheit;
   HORecHitCollection::const_iterator hoit;
   HFRecHitCollection::const_iterator hfit;

   if(firsttime)
   {
      int runnum = event.id().run();
      std::stringstream ostream;
      ostream << runnum;
      outFileName = ostream.str();
      outFileName += ".png";
      firsttime = false;
   }

   float energy = 0;
   for (hbheit  = hbherh->begin(); 
      hbheit != hbherh->end();
      hbheit++) {
      energy = 0;
      energy = hbheit->energy();
      HcalDetId id = hbheit->id();
      if (id.subdet() == 1) hbenergy->Fill(energy);
   }
   for (hbheit  = hbherh->begin();
      hbheit != hbherh->end();
      hbheit++) {
      energy = 0;
      energy = hbheit->energy();
      HcalDetId id = hbheit->id();
      if (id.subdet() == 2) heenergy->Fill(energy);
   }
   for (hoit  = horh->begin();
      hoit != horh->end();
      hoit++) {
      energy = 0;
      energy = hoit->energy();
      hoenergy->Fill(energy);
   }
   for (hfit  = hfrh->begin();
      hfit != hfrh->end();
      hfit++) {
      energy = 0;
      energy = hfit->energy();
      hfenergy->Fill(energy);
   }
}

DEFINE_FWK_MODULE(HcalPedestalsValidation);


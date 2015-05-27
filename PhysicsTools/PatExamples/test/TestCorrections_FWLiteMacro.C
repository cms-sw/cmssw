// -*- C++ -*-
//****Author -  Sudhir Malik - malik@fnal.gov *****//

  //*****NOTE: To successfully exeute this macro, include the following lines in your root logon file ************************//
    /*
      {
      gSystem->Load("libFWCoreFWLite.so");
      FWLiteEnabler::enable();
      gSystem->Load("libDataFormatsFWLite.so");
      gSystem->Load("libDataFormatsPatCandidates.so");
      gSystem->Load("libRooFit") ;
      using namespace RooFit ;
      cout << "loaded" << endl;
      }
    */
//*****************************//

  // CMS includes
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include <TH2.h>


#include "TFile.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TLegend.h"

#if !defined(__CINT__) && !defined(__MAKECINT__)

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "CondFormats/JetMETObjects/interface/CombinedJetCorrector.h"

  // these includes are needed to make the  "gSystem" commands below work
#include "TSystem.h"
#include "TROOT.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#endif


#include <iostream>
#include <vector>
#include <memory>


void JetCorrFWLite()
{
  
  
  // The input file MYCOPY.root  is available at
  //https://cms-service-sdtweb.web.cern.ch/cms-service-sdtweb/validation/physicstools/PATVALIDATION/MYCOPY.root
  
  TFile  *file = new TFile("MYCOPY.root");
  TFile  *outputfile    = new TFile("JetCorr.root","RECREATE");
  
  
  
  // Book those histograms for Jet stuff!
  TH1F * hist_jetMult = new TH1F( "jetMult", "jetMult", 100, 0, 50) ;
  TH1F * hist_jetPt = new TH1F( "jetPt", "jetPt", 100, 0, 200) ;
  TH1F * hist_jetEta = new TH1F( "jetEta", "jetEta", 100,-5, 5) ; 
  TH1F * hist_jetPhi = new TH1F( "jetPhi", "jetPhi", 100, -3.5, 3.5) ; 
  TH2F * hist_Mapping = new TH2F( "Mapping", "Mapping", 500, 0, 500, 500, 0, 500) ;
  TH2F * hist_Ratio = new TH2F( "Ratio", "Ratio", 500, 0, 500, 500, 0.8, 1.2) ;
  
  
  fwlite::Event ev(file);
  for( ev.toBegin();
       ! ev.atEnd();
       ++ev) {
    
    
    fwlite::Handle<vector<reco::CaloJet> > jetHandle;
    jetHandle.getByLabel(ev,"ak5CaloJets");
    
    
    
    ////////////////////////////////////////////////////////////////////////
    ////////////// Defining the L2L3L5L7JetCorrector ///////////////////////
    ////////////////////////////////////////////////////////////////////////
    
    string Levels1 = "L2:L3:L5:L7";
    string Tags1 = "900GeV_L2Relative_AK5Calo:900GeV_L3Absolute_AK5Calo:L5Flavor_IC5:L7Parton_IC5";
    string Options1 = "Flavor:gJ & Parton:gJ";
    CombinedJetCorrector *L2L3L5L7JetCorrector = new CombinedJetCorrector(Levels1,Tags1,Options1);
    
    ///////////////////////////////////////////////////////////////////////
      ////////////// Defining the L2L3JetCorrector //////////////////////////
      ///////////////////////////////////////////////////////////////////////
      
      string Levels = "L2:L3";
      string Tags = "900GeV_L2Relative_AK5Calo:900GeV_L3Absolute_AK5Calo";
      CombinedJetCorrector *L2L3JetCorrector = new CombinedJetCorrector(Levels,Tags);
      
      
      
      
      
      // Loop over the jets
      hist_jetMult->Fill(jetHandle->size()); 
      const vector<reco::CaloJet>::const_iterator kJetEnd = jetHandle->end();
      for (vector<reco::CaloJet>::const_iterator jetIter = jetHandle->begin();
           kJetEnd != jetIter; 
           ++jetIter) 
	{         
	  hist_jetPt ->Fill (jetIter->pt());
	  hist_jetEta->Fill (jetIter->eta());
	  hist_jetPhi->Fill (jetIter->phi());
	  ////////////////////Jet Correctiion Stuff ////////////
          double pt = jetIter->pt();
          double eta = jetIter->eta();
          double emf = jetIter->emEnergyFraction();
	  
          double L2L3scale = L2L3JetCorrector->getCorrection(pt,eta,emf); 
          double L2L3L5L7scale = L2L3L5L7JetCorrector->getCorrection(pt,eta,emf);
	  
          vector<double> L2L3factors = L2L3JetCorrector->getSubCorrections(pt,eta,emf);
          vector<double> L2L3L5L7factors = L2L3L5L7JetCorrector->getSubCorrections(pt,eta,emf);
	  
          cout<<"Pt = "<<pt<<", Eta = "<<eta<<", EMF = "<<emf<<endl;
          cout<<"L2L3correction = "<<L2L3scale<<", L2L3CorPt = "<<L2L3scale*pt<<endl;
          for(unsigned int i=0;i < L2L3factors.size();i++)
            cout<<L2L3factors[i]<<endl;
          cout<<"L2L3L5L7correction = "<<L2L3L5L7scale<<", L2L3L5L7CorPt = "<<L2L3L5L7scale*pt<<endl;
          for(unsigned int i=0;i< L2L3L5L7factors.size();i++)
            cout<<L2L3L5L7factors[i]<<endl;
          hist_Mapping->Fill (L2L3scale*pt,L2L3L5L7scale*pt);
          hist_Ratio->Fill (L2L3scale*pt,L2L3L5L7scale/L2L3scale);
	  cout << "L2L3scale*pt = " << L2L3scale*pt << endl;
	  cout << "L2L3L5L7scale*pt = " << L2L3L5L7scale*pt << endl;
          //////////////////// End of Jet Correctiion Stuff ////////////
	  
	} // for jetIter
      
      
      
  } // for event loop
  
  
  outputfile->cd();                
  outputfile->Write();
  outputfile->Close(); 
  
}


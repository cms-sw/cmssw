// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonCaloCompatibility
// 
/*

 Description: test track muon hypothesis using energy deposition in ECAL,HCAL,HO

*/
//
// Original Author:  Ingo Bloch
// $Id: MuonCaloCompatibility.cc,v 1.8 2010/06/02 15:50:51 andersj Exp $
//
//
#include "RecoMuon/MuonIdentification/interface/MuonCaloCompatibility.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void MuonCaloCompatibility::configure(const edm::ParameterSet& iConfig)
{
   MuonfileName_ = (iConfig.getParameter<edm::FileInPath>("MuonTemplateFileName")).fullPath();
   PionfileName_ = (iConfig.getParameter<edm::FileInPath>("PionTemplateFileName")).fullPath();
   muon_templates.reset( new TFile(MuonfileName_.c_str(),"READ") );
   pion_templates.reset( new TFile(PionfileName_.c_str(),"READ") );

   pion_em_etaEmi  = (TH2D*) pion_templates->Get("em_etaEmi");
   pion_had_etaEmi = (TH2D*) pion_templates->Get("had_etaEmi");
	       	
   pion_em_etaTmi  = (TH2D*) pion_templates->Get("em_etaTmi");
   pion_had_etaTmi = (TH2D*) pion_templates->Get("had_etaTmi");
		
   pion_em_etaB    = (TH2D*) pion_templates->Get("em_etaB");
   pion_had_etaB   = (TH2D*) pion_templates->Get("had_etaB");
   pion_ho_etaB    = (TH2D*) pion_templates->Get("ho_etaB");
   
   pion_em_etaTpl  = (TH2D*) pion_templates->Get("em_etaTpl");
   pion_had_etaTpl = (TH2D*) pion_templates->Get("had_etaTpl");
	       	
   pion_em_etaEpl  = (TH2D*) pion_templates->Get("em_etaEpl");
   pion_had_etaEpl = (TH2D*) pion_templates->Get("had_etaEpl");
		
   muon_em_etaEmi  = (TH2D*) muon_templates->Get("em_etaEmi");
   muon_had_etaEmi = (TH2D*) muon_templates->Get("had_etaEmi");
	       	
   muon_em_etaTmi  = (TH2D*) muon_templates->Get("em_etaTmi");
   muon_had_etaTmi = (TH2D*) muon_templates->Get("had_etaTmi");
	       	
   muon_em_etaB    = (TH2D*) muon_templates->Get("em_etaB");
   muon_had_etaB   = (TH2D*) muon_templates->Get("had_etaB");
   muon_ho_etaB    = (TH2D*) muon_templates->Get("ho_etaB");
	       	
   muon_em_etaTpl  = (TH2D*) muon_templates->Get("em_etaTpl");
   muon_had_etaTpl = (TH2D*) muon_templates->Get("had_etaTpl");
		
   muon_em_etaEpl  = (TH2D*) muon_templates->Get("em_etaEpl");
   muon_had_etaEpl = (TH2D*) muon_templates->Get("had_etaEpl");

   pbx = -1;
   pby = -1;
   pbz = -1;

   psx = -1;
   psy = -1;
   psz = -1;

   muon_compatibility = -1;

   use_corrected_hcal = true;
   use_em_special = true;
   isConfigured_ = true;
}

bool MuonCaloCompatibility::accessing_overflow( TH2D* histo, double x, double y ) {
  bool access = false;

  if( histo->GetXaxis()->FindBin(x) == 0 || 
      histo->GetXaxis()->FindBin(x) > histo->GetXaxis()->GetNbins() ) {
    access = true;
  }
  if( histo->GetYaxis()->FindBin(y) == 0 || 
      histo->GetYaxis()->FindBin(y) > histo->GetYaxis()->GetNbins() ) {
    access = true;
  }
  return access;
}

double MuonCaloCompatibility::evaluate( const reco::Muon& amuon ) {
  if (! isConfigured_) {
     edm::LogWarning("MuonIdentification") << "MuonCaloCompatibility is not configured! Nothing is calculated.";
     return -9999;
  }
   
  double eta = 0.;
  double p   = 0.;
  double em  = 0.;
  double had = 0.;
  double ho  = 0.;

  // had forgotten this reset in previous versions 070409
  pbx = 1.;
  pby = 1.;
  pbz = 1.;

  psx = 1.;
  psy = 1.;
  psz = 1.;

  muon_compatibility = -1.;
  
  pion_template_em   = NULL;
  muon_template_em   = NULL;
  
  pion_template_had  = NULL;
  muon_template_had  = NULL;
  
  pion_template_ho   = NULL;
  muon_template_ho   = NULL;
  
  // 071002: Get either tracker track, or SAmuon track.
  // CaloCompatibility templates may have to be specialized for 
  // the use with SAmuons, currently just using the ones produced
  // using tracker tracks. 
  const reco::Track* track = 0;
  if ( ! amuon.track().isNull() ) {
    track = amuon.track().get();
  }
  else {
    if ( ! amuon.standAloneMuon().isNull() ) {
      track = amuon.standAloneMuon().get();
    }
    else {
      throw cms::Exception("FatalError") << "Failed to fill muon id calo_compatibility information for a muon with undefined references to tracks"; 
    }
  }

  if( !use_corrected_hcal ) { // old eta regions, uncorrected energy
    eta = track->eta();
    p   = track->p(); 

    // new 070904: Set lookup momentum to 1999.9 if larger than 2 TeV. 
    // Though the templates were produced with p<2TeV, we believe that
    // this approximation should be roughly valid. A special treatment
    // for >1 TeV muons is advisable anyway :)
    if( p>=2000. ) p = 1999.9;

    //    p   = 10./sin(track->theta()); // use this for templates < 1_5
    if( use_em_special ) {
      if( amuon.calEnergy().em == 0. )    em  = -5.;
      else em  = amuon.calEnergy().em;
    }
    else {
      em  = amuon.calEnergy().em;
    }
    had = amuon.calEnergy().had;
    ho  = amuon.calEnergy().ho;
  }
  else {
    eta = track->eta();
    p   = track->p();
    
    // new 070904: Set lookup momentum to 1999.9 if larger than 2 TeV. 
    // Though the templates were produced with p<2TeV, we believe that
    // this approximation should be roughly valid. A special treatment
    // for >1 TeV muons is advisable anyway :)
    if( p>=2000. ) p = 1999.9;

    //    p   = 10./sin(track->theta());  // use this for templates < 1_5
    // hcal energy is now done where we get the template histograms (to use corrected cal energy)!
    //     had = amuon.calEnergy().had;
    if( use_em_special ) {
      if( amuon.calEnergy().em == 0. )    em  = -5.;
      else em  = amuon.calEnergy().em;
    }
    else {
      em  = amuon.calEnergy().em;
    }
    ho  = amuon.calEnergy().ho;
  }


  // Skip everyting and return "I don't know" (i.e. 0.5) for uncovered regions:
  //  if( p < 0. || p > 500.) return 0.5; // removed 500 GeV cutoff 070817 after updating the tempates (v2_0) to have valid entried beyond 500 GeV
  if( p < 0. ) return 0.5; // return "unknown" for unphysical momentum input.
  if( fabs(eta) >  2.5 ) return 0.5; 
  // temporary fix for low association efficiency:
  // set caloCompatibility to 0.12345 for tracks
  // which have 0 energy in BOTH ecal and hcal
  if( amuon.calEnergy().had == 0.0 && amuon.calEnergy().em == 0.0 ) return 0.12345; 

  //  std::cout<<std::endl<<"Input values are: "<<eta <<" "<< p <<" "<< em <<" "<< had <<" "<< ho;

  //  depending on the eta, choose correct histogram: (now all for barrel):
  // bad! eta range has to be syncronised with choice for histogram... should be read out from the histo file somehow... 070322
  if(42 != 42) { // old eta ranges and uncorrected hcal energy
    if(eta <= -1.4) {
      //    std::cout<<"Emi"<<std::endl;
      pion_template_em  = pion_em_etaEmi;
      pion_template_had = pion_had_etaEmi;
      muon_template_em  = muon_em_etaEmi;
      muon_template_had = muon_had_etaEmi;
    }
    else if(eta > -1.4 && eta <= -1.31) {
      //    std::cout<<"Tmi"<<std::endl;
      pion_template_em  = pion_em_etaTmi;
      pion_template_had = pion_had_etaTmi;
      muon_template_em  = muon_em_etaTmi;
      muon_template_had = muon_had_etaTmi;
    }
    else if(eta > -1.31 && eta <= 1.31) {
      //    std::cout<<"B"<<std::endl;
      pion_template_em  = pion_em_etaB;
      pion_template_had = pion_had_etaB;
      pion_template_ho  = pion_ho_etaB;
      muon_template_em  = muon_em_etaB;
      muon_template_had = muon_had_etaB;
      muon_template_ho  = muon_ho_etaB;
    }
    else if(eta > 1.31 && eta <= 1.4) {
      //    std::cout<<"Tpl"<<std::endl;
      pion_template_em  = pion_em_etaTpl;
      pion_template_had = pion_had_etaTpl;
      muon_template_em  = muon_em_etaTpl;
      muon_template_had = muon_had_etaTpl;
    }
    else if(eta > 1.4) {
      //    std::cout<<"Epl"<<std::endl;
      pion_template_em  = pion_em_etaEpl;
      pion_template_had = pion_had_etaEpl;
      muon_template_em  = muon_em_etaEpl;
      muon_template_had = muon_had_etaEpl;
    }
    else {
      LogTrace("MuonIdentification")<<"Some very weird thing happened in MuonCaloCompatibility::evaluate - go figure ;) ";
      return -999;
    }
  }
  else if( 42 == 42 ) { // new eta bins, corrected hcal energy
    if(  track->eta() >  1.27  ) {
      // 	had_etaEpl ->Fill(muon->track().get()->p(),1.8/2.2*muon->calEnergy().had );
      if(use_corrected_hcal)	had = 1.8/2.2*amuon.calEnergy().had;
      else	had = amuon.calEnergy().had;
      pion_template_had = pion_had_etaEpl;
      muon_template_had = muon_had_etaEpl;
    }
    if( track->eta() <=  1.27  && track->eta() >  1.1 ) {
      // 	had_etaTpl ->Fill(muon->track().get()->p(),(1.8/(-2.2*muon->track().get()->eta()+5.5))*muon->calEnergy().had );
      if(use_corrected_hcal)	had = (1.8/(-2.2*track->eta()+5.5))*amuon.calEnergy().had;
      else 	had = amuon.calEnergy().had;
      pion_template_had  = pion_had_etaTpl;
      muon_template_had  = muon_had_etaTpl;
    }
    if( track->eta() <=  1.1 && track->eta() > -1.1 ) {
      // 	had_etaB   ->Fill(muon->track().get()->p(),sin(muon->track().get()->theta())*muon->calEnergy().had );
      if(use_corrected_hcal)	had = sin(track->theta())*amuon.calEnergy().had;
      else 	had = amuon.calEnergy().had;
      pion_template_had  = pion_had_etaB;
      muon_template_had  = muon_had_etaB;
    }
    if( track->eta() <= -1.1 && track->eta() > -1.27 ) {
      // 	had_etaTmi ->Fill(muon->track().get()->p(),(1.8/(-2.2*muon->track().get()->eta()+5.5))*muon->calEnergy().had );
      if(use_corrected_hcal)	had = (1.8/(2.2*track->eta()+5.5))*amuon.calEnergy().had;
      else 	had = amuon.calEnergy().had;
      pion_template_had = pion_had_etaTmi;
      muon_template_had = muon_had_etaTmi;
    }
    if( track->eta() <= -1.27 ) {
      // 	had_etaEmi ->Fill(muon->track().get()->p(),1.8/2.2*muon->calEnergy().had );
      if(use_corrected_hcal)	had = 1.8/2.2*amuon.calEnergy().had;
      else 	had = amuon.calEnergy().had;
      pion_template_had = pion_had_etaEmi;
      muon_template_had = muon_had_etaEmi;
    }
    
    // just two eta regions for Ecal (+- 1.479 for barrel, else for rest), no correction:

    //    std::cout<<"We have a muon with an eta of: "<<track->eta()<<std::endl;

    if(  track->eta() >  1.479  ) {
      // 	em_etaEpl  ->Fill(muon->track().get()->p(),muon->calEnergy().em   );
      // 	//	em_etaTpl  ->Fill(muon->track().get()->p(),muon->calEnergy().em   );
      ////      em  = amuon.calEnergy().em;
      pion_template_em  = pion_em_etaEpl;
      muon_template_em  = muon_em_etaEpl;
    }
    if( track->eta() <=  1.479 && track->eta() > -1.479 ) {
      // 	em_etaB    ->Fill(muon->track().get()->p(),muon->calEnergy().em   );
      ////      em  = amuon.calEnergy().em;
      pion_template_em  = pion_em_etaB;
      muon_template_em  = muon_em_etaB;
    }
    if( track->eta() <= -1.479 ) {
      // 	//	em_etaTmi  ->Fill(muon->track().get()->p(),muon->calEnergy().em   );
      // 	em_etaEmi  ->Fill(muon->track().get()->p(),muon->calEnergy().em   );
      ////      em  = amuon.calEnergy().em;
      pion_template_em  = pion_em_etaEmi;
      muon_template_em  = muon_em_etaEmi;
    }
    
    // just one barrel eta region for the HO, no correction
    //    if( track->eta() < 1.4 && track->eta() > -1.4 ) { // experimenting now...
    if( track->eta() < 1.28 && track->eta() > -1.28 ) {
      // 	ho_etaB    ->Fill(muon->track().get()->p(),muon->calEnergy().ho   );
      ////      ho  = amuon.calEnergy().ho;
      pion_template_ho  = pion_ho_etaB;
      muon_template_ho  = muon_ho_etaB;
    }

    
  }


  if( 42 != 42 ) { // check validity of input template histos and input variables"
    pion_template_em ->ls();
    pion_template_had->ls();
    if(pion_template_ho)   pion_template_ho ->ls();
    muon_template_em ->ls();
    muon_template_had->ls();
    if(muon_template_ho)   muon_template_ho ->ls();

    LogTrace("MuonIdentification")<<"Input variables: eta    p     em     had    ho "<<"\n"
	     <<eta<<" "<<p<<" "<<em<<" "<<had<<" "<<ho<<" "<<"\n"
	     <<"cal uncorr:    em     had    ho "<<"\n"
	     <<eta<<" "<<p<<" "<<amuon.calEnergy().em<<" "<<amuon.calEnergy().had<<" "<<amuon.calEnergy().ho;
  }
  

  //  Look up Compatibility by, where x is p and y the energy. 
  //  We have a set of different histograms for different regions of eta.

  // need error meassage in case the template histos are missing / the template file is not present!!! 070412

  if( pion_template_em )  { // access ecal background template
    if( accessing_overflow( pion_template_em, p, em ) ) {
      pbx = 1.;
      psx = 1.;
      LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for ecal - defaulting signal and background  ";
      LogTrace("MuonIdentification")<<"            // template value to 1. "<<pion_template_em->GetName()<<" e: "<<em<<" p: "<<p;
    }
    else pbx =  pion_template_em->GetBinContent(  pion_template_em->GetXaxis()->FindBin(p), pion_template_em->GetYaxis()->FindBin(em) );
  }
  if( pion_template_had ) { // access hcal background template
    if( accessing_overflow( pion_template_had, p, had ) ) {
      pby = 1.;
      psy = 1.;
      LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for hcal - defaulting signal and background  ";
      LogTrace("MuonIdentification")<<"            // template value to 1. "<<pion_template_had->GetName()<<" e: "<<had<<" p: "<<p;
    }
    else pby =  pion_template_had->GetBinContent(  pion_template_had->GetXaxis()->FindBin(p), pion_template_had->GetYaxis()->FindBin(had) );
  }
  if(pion_template_ho) { // access ho background template
    if( accessing_overflow( pion_template_ho, p, ho ) ) {
      pbz = 1.;
      psz = 1.;
      LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for ho   - defaulting signal and background  ";
      LogTrace("MuonIdentification")<<"            // template value to 1. "<<pion_template_ho->GetName()<<" e: "<<em<<" p: "<<p; 
    }
    else pbz =  pion_template_ho->GetBinContent(  pion_template_ho->GetXaxis()->FindBin(p), pion_template_ho->GetYaxis()->FindBin(ho) );
  }


  if( muon_template_em )  { // access ecal background template
    if( accessing_overflow( muon_template_em, p, em ) ) {
      psx = 1.;
      pbx = 1.;
      LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for ecal - defaulting signal and background  ";
      LogTrace("MuonIdentification")<<"            // template value to 1. "<<muon_template_em->GetName()<<" e: "<<em<<" p: "<<p;
    }
    else psx =  muon_template_em->GetBinContent(  muon_template_em->GetXaxis()->FindBin(p), muon_template_em->GetYaxis()->FindBin(em) );
  }
  if( muon_template_had ) { // access hcal background template
    if( accessing_overflow( muon_template_had, p, had ) ) {
      psy = 1.;
      pby = 1.;
      LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for hcal - defaulting signal and background  ";
      LogTrace("MuonIdentification")<<"            // template value to 1. "<<muon_template_had->GetName()<<" e: "<<had<<" p: "<<p;
    }
    else psy =  muon_template_had->GetBinContent(  muon_template_had->GetXaxis()->FindBin(p), muon_template_had->GetYaxis()->FindBin(had) );
  }
  if(muon_template_ho) { // access ho background template
    if( accessing_overflow( muon_template_ho, p, ho ) ) {
      psz = 1.;
      pbz = 1.;
       LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for ho   - defaulting signal and background  ";
       LogTrace("MuonIdentification")<<"            // template value to 1. "<<muon_template_ho->GetName()<<" e: "<<ho<<" p: "<<p;
    }
    else psz =  muon_template_ho->GetBinContent(  muon_template_ho->GetXaxis()->FindBin(p), muon_template_ho->GetYaxis()->FindBin(ho) );
  }

  // erm - what is this?!?! How could the HO probability be less than 0????? Do we want this line!?!?
  if(psz <= 0.) psz = 1.;
  if(pbz <= 0.) pbz = 1.;

  // Protection agains empty bins - set cal part to neutral if the bin of the template is empty 
  // (temporary fix, a proper extrapolation would be better)
  if (psx == 0. || pbx == 0.) {
    psx = 1.;
    pbx = 1.;
  } 
  if (psy == 0. || pby == 0.) {
    psy = 1.;
    pby = 1.;
  } 
  if (psz == 0. || pbz == 0.) {
    psz = 1.;
    pbz = 1.;
  }

  // There are two classes of events that deliver 0 for the hcal or ho energy:
  // 1) the track momentum is so low that the extrapolation tells us it should not have reached the cal
  // 2) the crossed cell had an reading below the readout cuts.
  // The 2nd case discriminates between muons and hadrons, the 1st not. Thus for the time being, 
  // we set the returned ps and pb to 0 for these cases.
  // We need to have a return value different from 0 for the 1st case in the long run.
  if ( had == 0.0 ) {
    psy = 1.;
    pby = 1.;
  } 
  if ( ho == 0.0 ) {
    psz = 1.;
    pbz = 1.;
  }
  

  // Set em to neutral if no energy in em or negative energy measured. 
  // (These cases might indicate problems in the ecal association or readout?! The only 
  // hint so far: for critical eta region (eta in [1.52, 1.64]) have negative em values.)
  if( em <= 0. && !use_em_special ) {
    pbx = 1.;
    psx = 1.;
  }

  if( (psx*psy*psz+pbx*pby*pbz) > 0. ) muon_compatibility = psx*psy*psz / (psx*psy*psz+pbx*pby*pbz);
  else {
    LogTrace("MuonIdentification")<<"Divide by 0 - defaulting consistency to 0.5 (neutral)!!";
    muon_compatibility = 0.5;
    LogTrace("MuonIdentification")<<"Input variables: eta    p     em     had    ho "<<"\n"
	     <<eta<<" "<<p<<" "<<em<<" "<<had<<" "<<ho<<" "<<"\n"
	     <<"cal uncorr:    em     had    ho "<<"\n"
	     <<eta<<" "<<p<<" "<<amuon.calEnergy().em<<" "<<amuon.calEnergy().had<<" "<<amuon.calEnergy().ho;
  }
  return muon_compatibility;
}

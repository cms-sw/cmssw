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
// $Id: MuonCaloCompatibility.cc,v 1.6 2010/01/21 01:39:00 slava77 Exp $
//
//
#include "RecoMuon/MuonIdentification/interface/MuonCaloCompatibility.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//need to move to package
#include "RecoMuon/MuonIdentification/interface/MuonHOAcceptance.h"

void MuonCaloCompatibility::configure(const edm::ParameterSet& iConfig)
{
   MuonfileName_ = (iConfig.getParameter<edm::FileInPath>("MuonTemplateFileName")).fullPath();
   PionfileName_ = (iConfig.getParameter<edm::FileInPath>("PionTemplateFileName")).fullPath();

   delta_eta = iConfig.getParameter<double>("delta_eta");
   delta_phi = iConfig.getParameter<double>("delta_phi");
   allSiPMHO = iConfig.getParameter<bool>("allSiPMHO");

   // std::cout << " muon file: " << MuonfileName_ 
   // 	     << " pion file: " << PionfileName_ << std::endl;
   muon_templates.reset( new TFile(MuonfileName_.c_str(),"READ") );
   pion_templates.reset( new TFile(PionfileName_.c_str(),"READ") );

   pion_em_etaEmi  = (TH2D*) pion_templates->Get("em_etaEmi");
   pion_had_etaEmi = (TH2D*) pion_templates->Get("had_etaEmi");
	       	
   pion_em_etaTmi  = (TH2D*) pion_templates->Get("em_etaTmi");
   pion_had_etaTmi = (TH2D*) pion_templates->Get("had_etaTmi");
		
   pion_em_etaB    = (TH2D*) pion_templates->Get("em_etaB");
   pion_had_etaB   = (TH2D*) pion_templates->Get("had_etaB");
   pion_ho_etaB0   = (TH2D*) pion_templates->Get("ho_etaB0");
   pion_ho_etaBpl  = (TH2D*) pion_templates->Get("ho_etaBpl");
   pion_ho_etaBmi  = (TH2D*) pion_templates->Get("ho_etaBmi");
   pion_ho_SiPMs  =  (TH2D*) pion_templates->Get("ho_SiPMs");
   
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
   muon_ho_etaB0   = (TH2D*) muon_templates->Get("ho_etaB0");
   muon_ho_etaBpl  = (TH2D*) muon_templates->Get("ho_etaBpl");
   muon_ho_etaBmi  = (TH2D*) muon_templates->Get("ho_etaBmi");
   muon_ho_SiPMs   = (TH2D*) muon_templates->Get("ho_SiPMs");
	       	
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

double MuonCaloCompatibility::evaluate( const reco::Muon& amuon, 
					CaloCompatType ty ) {
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
    ho  = amuon.calEnergy().hoMax;
  }
  else {
    eta = track->eta();
    p   = track->p();
    
    // new 070904: Set lookup momentum to 1999.9 if larger than 2 TeV. 
    // Though the templates were produced with p<2TeV, we believe that
    // this approximation should be roughly valid. A special treatment
    // for >1 TeV muons is advisable anyway :)
    if( p>=2000. ) p = 1999.9;

    if( use_em_special ) {
      if( amuon.calEnergy().em == 0. )    em  = -5.;
      else em  = amuon.calEnergy().em;
    }
    else {
      em  = amuon.calEnergy().em;
    }
    // hcal energy is now done where we get the template histograms 
    // (to use corrected cal energy)!
  }


  // Skip everyting and return "I don't know" (i.e. 0.5) for uncovered regions:
  //  if( p < 0. || p > 500.) return 0.5; // removed 500 GeV cutoff 070817 after updating the tempates (v2_0) to have valid entried beyond 500 GeV
  if( p < 0. ) return 0.5; // return "unknown" for unphysical momentum input.
  if( fabs(eta) >  2.5 ) return 0.5; 
  // temporary fix for low association efficiency:
  // set caloCompatibility to 0.12345 for tracks
  // which have 0 energy in BOTH ecal and hcal
  // if( amuon.calEnergy().had == 0.0 && amuon.calEnergy().em == 0.0 ) return 0.12345; 

  //  std::cout<<std::endl<<"Input values are: "<<eta <<" "<< p <<" "<< em <<" "<< had <<" "<< ho;

  //  depending on the eta, choose correct histogram: (now all for barrel):
  // bad! eta range has to be syncronised with choice for histogram... should be read out from the histo file somehow... 070322
  if (!amuon.calEnergy().hcal_id.null()) {
    if(  track->eta() >  1.27  ) {
      if(use_corrected_hcal)
	had = 1.8/2.2*amuon.calEnergy().had;
      else
	had = amuon.calEnergy().had;
      pion_template_had = pion_had_etaEpl;
      muon_template_had = muon_had_etaEpl;
    }
    if( track->eta() <=  1.27  && track->eta() >  1.1 ) {
      if(use_corrected_hcal)	
	had = (1.8/(-2.2*track->eta()+5.5))*amuon.calEnergy().had;
      else
	had = amuon.calEnergy().had;
      pion_template_had  = pion_had_etaTpl;
      muon_template_had  = muon_had_etaTpl;
    }
    if( track->eta() <=  1.1 && track->eta() > -1.1 ) {
      if(use_corrected_hcal)
	had = sin(track->theta())*amuon.calEnergy().had;
      else
	had = amuon.calEnergy().had;
      pion_template_had  = pion_had_etaB;
      muon_template_had  = muon_had_etaB;
    }
    if( track->eta() <= -1.1 && track->eta() > -1.27 ) {
      if(use_corrected_hcal)
	had = (1.8/(2.2*track->eta()+5.5))*amuon.calEnergy().had;
      else
	had = amuon.calEnergy().had;
      pion_template_had = pion_had_etaTmi;
      muon_template_had = muon_had_etaTmi;
    }
    if( track->eta() <= -1.27 ) {
      if(use_corrected_hcal)
	had = 1.8/2.2*amuon.calEnergy().had;
      else
	had = amuon.calEnergy().had;
      pion_template_had = pion_had_etaEmi;
      muon_template_had = muon_had_etaEmi;
    }
  }
  // just two eta regions for Ecal (+- 1.479 for barrel, else for rest), no correction:

  if(  track->eta() >  1.479  ) {
    pion_template_em  = pion_em_etaEpl;
    muon_template_em  = muon_em_etaEpl;
  }
  if( track->eta() <=  1.479 && track->eta() > -1.479 ) {
    pion_template_em  = pion_em_etaB;
    muon_template_em  = muon_em_etaB;
  }
  if( track->eta() <= -1.479 ) {
    pion_template_em  = pion_em_etaEmi;
    muon_template_em  = muon_em_etaEmi;
  }
    
  double hoeta = amuon.calEnergy().ho_position.Eta();
  double hophi = amuon.calEnergy().ho_position.Phi();
  unsigned int muAccept = 0;
  if (!amuon.calEnergy().ho_id.null())
    muAccept += 1;
  if ( MuonHOAcceptance::inGeomAccept(hoeta,hophi,delta_eta,delta_phi) )
    muAccept += 10;
  if (MuonHOAcceptance::inNotDeadGeom(hoeta,hophi,delta_eta,delta_phi))
    muAccept += 100;
  if (MuonHOAcceptance::inSiPMGeom(hoeta,hophi,0., 0.))
    muAccept += 1000;
  if (muAccept%1000==111) {
    ho  = sin(track->theta())*amuon.calEnergy().hoMax;
    if ( !allSiPMHO && ( (muAccept/1000)%10==1 ) ) {
      pion_template_ho = pion_ho_SiPMs;
      muon_template_ho = muon_ho_SiPMs;
    }
    else {
      // Ring 0 eta region of HO
      if( track->eta() < 0.348 && track->eta() > -0.348 ) {
	pion_template_ho  = pion_ho_etaB0;
	muon_template_ho  = muon_ho_etaB0;
      }
    
      // barrel rings 1 and 2 plus eta regionn of HO
      if( track->eta() < 1.3 && track->eta() >= 0.348 ) {
	pion_template_ho  = pion_ho_etaBpl;
	muon_template_ho  = muon_ho_etaBpl;
      }

      // barrel rings 1 and 2 minus eta region of HO
      if( track->eta() <= -0.348  && track->eta() > -1.3 ) {
	pion_template_ho  = pion_ho_etaBmi;
	muon_template_ho  = muon_ho_etaBmi;
      }
    }
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
    else 
      pbx = pion_template_em->GetBinContent( pion_template_em->GetXaxis()->FindBin(p), pion_template_em->GetYaxis()->FindBin(em) );
  }
  if( pion_template_had ) { // access hcal background template
    if( accessing_overflow( pion_template_had, p, had ) ) {
      pby = 1.;
      psy = 1.;
      LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for hcal - defaulting signal and background  ";
      LogTrace("MuonIdentification")<<"            // template value to 1. "<<pion_template_had->GetName()<<" e: "<<had<<" p: "<<p;
    }
    else 
      pby = pion_template_had->GetBinContent( pion_template_had->GetXaxis()->FindBin(p), pion_template_had->GetYaxis()->FindBin(had) );
  }
  if(pion_template_ho) { // access ho background template
    if( accessing_overflow( pion_template_ho, p, ho ) ) {
      pbz = 1.;
      psz = 1.;
      LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for ho   - defaulting signal and background  ";
      LogTrace("MuonIdentification")<<"            // template value to 1. "<<pion_template_ho->GetName()<<" e: "<<em<<" p: "<<p; 
    }
    else 
      pbz = pion_template_ho->GetBinContent( pion_template_ho->GetXaxis()->FindBin(p), pion_template_ho->GetYaxis()->FindBin(ho) );
  }


  if( muon_template_em )  { // access ecal signal template
    if( accessing_overflow( muon_template_em, p, em ) ) {
      psx = 1.;
      pbx = 1.;
      LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for ecal - defaulting signal and background  ";
      LogTrace("MuonIdentification")<<"            // template value to 1. "<<muon_template_em->GetName()<<" e: "<<em<<" p: "<<p;
    }
    else 
      psx = muon_template_em->GetBinContent( muon_template_em->GetXaxis()->FindBin(p), muon_template_em->GetYaxis()->FindBin(em) );
  }
  if( muon_template_had ) { // access hcal signal template
    if( accessing_overflow( muon_template_had, p, had ) ) {
      psy = 1.;
      pby = 1.;
      LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for hcal - defaulting signal and background  ";
      LogTrace("MuonIdentification")<<"            // template value to 1. "<<muon_template_had->GetName()<<" e: "<<had<<" p: "<<p;
    }
    else 
      psy = muon_template_had->GetBinContent( muon_template_had->GetXaxis()->FindBin(p), muon_template_had->GetYaxis()->FindBin(had) );
  }
  if(muon_template_ho) { // access ho signal template
    if( accessing_overflow( muon_template_ho, p, ho ) ) {
      psz = 1.;
      pbz = 1.;
       LogTrace("MuonIdentification")<<"            // Message: trying to access overflow bin in MuonCompatibility template for ho   - defaulting signal and background  ";
       LogTrace("MuonIdentification")<<"            // template value to 1. "<<muon_template_ho->GetName()<<" e: "<<ho<<" p: "<<p;
    }
    else 
      psz = muon_template_ho->GetBinContent( muon_template_ho->GetXaxis()->FindBin(p), muon_template_ho->GetYaxis()->FindBin(ho) );
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
  // if ( had == 0.0 ) {
  //   psy = 1.;
  //   pby = 1.;
  // } 
  // if ( ho == 0.0 ) {
  //   psz = 1.;
  //   pbz = 1.;
  // }
  

  // Set em to neutral if no energy in em or negative energy measured. 
  // (These cases might indicate problems in the ecal association or readout?! The only 
  // hint so far: for critical eta region (eta in [1.52, 1.64]) have negative em values.)
  if( em <= 0. && !use_em_special ) {
    pbx = 1.;
    psx = 1.;
  }

  if (ty == HOCalo) {
    if ( (muon_template_ho) && (pion_template_ho) && (psz+pbz > 0) )
      muon_compatibility = psz/(psz+pbz);
    else
      muon_compatibility = -1.;
  }
  else if (ty == HadCalo) {
    if ( (psy*psz + pby*pbz > 0) && 
	 (muon_template_had)  && (pion_template_had) )
      muon_compatibility = psy*psz/(psy*psz + pby*pbz);
    else
      muon_compatibility = -1.;
  }
  else if ( (ty == AllCalo) && ((psx*psy*psz+pbx*pby*pbz) > 0.) ) {
    muon_compatibility = psx*psy*psz / (psx*psy*psz+pbx*pby*pbz);
  }
  else {
    LogTrace("MuonIdentification")<<"Divide by 0 - defaulting consistency to 0.5 (neutral)!!";
    muon_compatibility = 0.5;
    LogTrace("MuonIdentification")<<"Input variables: eta    p     em     had    ho "<<"\n"
	     <<eta<<" "<<p<<" "<<em<<" "<<had<<" "<<ho<<" "<<"\n"
	     <<"cal uncorr:    em     had    ho "<<"\n"
	     <<eta<<" "<<p<<" "<<amuon.calEnergy().em<<" "<<amuon.calEnergy().had<<" "<<amuon.calEnergy().hoMax;
  }
  return muon_compatibility;
}

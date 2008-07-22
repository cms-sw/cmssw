/** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author M. Vander Donckt, J. Klukas  (copied from J. Alcaraz)
 */

#include "HLTriggerOffline/Muon/interface/HLTMuonGenericRate.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

// Collaborating Class Header
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"
#include "TDirectory.h"
#include "TH1F.h"
#include <iostream>
using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;
typedef std::vector< edm::ParameterSet > Parameters;

/// Constructor
HLTMuonGenericRate::HLTMuonGenericRate(const ParameterSet& pset, int index)
{
  useMuonFromGenerator = pset.getParameter<bool>("UseMuonFromGenerator");
  useMuonFromReco = pset.getParameter<bool>("UseMuonFromReco");
  if ( useMuonFromGenerator ) 
    theGenLabel = pset.getUntrackedParameter<InputTag>("GenLabel");
  if ( useMuonFromReco )
    theRecoLabel = pset.getUntrackedParameter<InputTag>("RecoLabel");
  
  Parameters TriggerLists = pset.getParameter<Parameters>("TriggerCollection");
  ParameterSet thisTrigger = TriggerLists[index];
  theL1CollectionLabel = 
    thisTrigger.getParameter<InputTag>("L1CollectionLabel");
  theHLTCollectionLabels = 
    thisTrigger.getParameter<std::vector<InputTag> >("HLTCollectionLabels");
  theL1ReferenceThreshold = 
    thisTrigger.getParameter<double>("L1ReferenceThreshold");    
  theHLTReferenceThreshold = 
    thisTrigger.getParameter<double>("HLTReferenceThreshold");    
  theNumberOfObjects = 
    thisTrigger.getParameter<unsigned int>("NumberOfObjects");

  theNSigmas = pset.getUntrackedParameter<std::vector<double> >("NSigmas90");
  theCrossSection = pset.getParameter<double>("CrossSection");

  // Convert it already into /nb/s
  theLuminosity = pset.getUntrackedParameter<double>("Luminosity",1.e+32)*1.e-33;

  thePtMin = pset.getUntrackedParameter<double>("PtMin",0.);
  thePtMax = pset.getUntrackedParameter<double>("PtMax",40.);
  theNbins = pset.getUntrackedParameter<unsigned int>("Nbins",40);
  
  theNumberOfEvents = 0;
  theNumberOfL1Events = 0;

  dbe_ = 0 ;
  if (pset.getUntrackedParameter<bool>("DQMStore", false)) {
    dbe_ = Service<DQMStore>().operator->();
    dbe_->setVerbose(0);
  }

  if ( pset.getUntrackedParameter<bool>("disableROOToutput", false) ) 
    theRootFileName="";
  else 
    theRootFileName = pset.getUntrackedParameter<std::string>("RootFileName");

  if ( dbe_ != NULL ) {
    dbe_->cd();
    dbe_->setCurrentFolder("HLT/Muon");
    dbe_->setCurrentFolder("HLT/Muon/RateEfficiencies");
    dbe_->setCurrentFolder("HLT/Muon/Distributions");
  }
}



/// Destructor
HLTMuonGenericRate::~HLTMuonGenericRate(){
}



void HLTMuonGenericRate::analyze(const Event & event ){
  thisEventWeight = 1;
  NumberOfEvents->Fill(++theNumberOfEvents); // Sets ME<int> NumberOfEvents
  LogTrace( "HLTMuonVal" ) << "In analyze for L1 trigger " << 
    theL1CollectionLabel << " Event:" << theNumberOfEvents;  

  // Get the muon with maximum pt at both generator and reconstruction levels 

  bool foundRefGenMuon = false;
  bool foundRefRecMuon = false;

  double genMuonPt = -1;
  double recMuonPt = -1;

  if (useMuonFromGenerator) {
    Handle<HepMCProduct> genProduct;
    event.getByLabel(theGenLabel, genProduct);
    if ( genProduct.failedToGet() ){
      LogWarning("HLTMuonVal") << "No generator input to compare to";
      useMuonFromGenerator = false;
    } else {
      evt = genProduct->GetEvent();
      for ( HepMC::GenEvent::particle_const_iterator iParticle = 
	    evt->particles_begin(); 
	    iParticle != evt->particles_end(); ++iParticle ) {
	if ( abs((*iParticle)->pdg_id()) == 13 && 
	     (*iParticle)->status() == 1 ) {
	  float pt = (*iParticle)->momentum().perp();
	  hMCetanor->Fill((*iParticle)->momentum().eta());
	  hMCphinor->Fill((*iParticle)->momentum().phi());
	  if ( pt > genMuonPt) {
	    foundRefGenMuon = true;
	    genMuonPt = pt;
	  }
	}
      }
    } 
  }

  Handle<reco::TrackCollection> muTracks;
  if (useMuonFromReco) {
    reco::TrackCollection::const_iterator muon;
    event.getByLabel(theRecoLabel.label(), muTracks);    
    if  ( muTracks.failedToGet() ) {
      LogWarning("HLTMuonVal") << "No reco tracks to compare to";
      useMuonFromReco = false;
    } else {
      for ( muon = muTracks->begin(); muon != muTracks->end(); ++muon ) {
	float pt = muon->pt();
	hRECOetanor->Fill( muon->eta() );
	hRECOphinor->Fill( muon->phi() );
	if ( pt > recMuonPt ) {
	  foundRefRecMuon = true;
	  recMuonPt = pt;
	}
      }
    }
  } 
  
  if ( foundRefGenMuon ) hMCptnor->Fill(genMuonPt, thisEventWeight);
  if ( foundRefRecMuon ) hRECOptnor->Fill(recMuonPt, thisEventWeight);

  // Get the Trigger collection
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByLabel("hltTriggerSummaryRAW", triggerObj); 

  if(!triggerObj.isValid()) { 
    edm::LogWarning("HLTMuonVal") << 
      "RAW-type HLT results not found, skipping event";
    return;
  } else {
     const size_type numFilterObjects(triggerObj->size());
     LogTrace("HLTMuonVal") << 
       "Used Processname: " << triggerObj->usedProcessName();
     LogTrace("HLTMuonVal") << 
       "Number of TriggerFilterObjects: " << numFilterObjects;
     LogTrace("HLTMuonVal") << "The TriggerFilterObjects: #, tag" ;
     for ( size_type i = 0; i != numFilterObjects; ++i )
      LogTrace("HLTMuonVal") << i << " " << triggerObj->filterTag(i).encode();
  }

  vector<L1MuonParticleRef> l1Cands;
  LogTrace("HLTMuonVal")<<"TriggerObject Size="<<triggerObj->size();
  if ( triggerObj->filterIndex(theL1CollectionLabel)>=triggerObj->size() ){
    LogTrace("HLTMuonVal") << "No L1 Collection with label "  
                           << theL1CollectionLabel;
    return;
  }
  triggerObj->getObjects(triggerObj->filterIndex(theL1CollectionLabel), 
			 81, l1Cands);
  NumberOfL1Events->Fill(++theNumberOfL1Events);

  // Get the HLT collections
  unsigned hltsize = theHLTCollectionLabels.size();
  vector< vector<RecoChargedCandidateRef> > hltCands(hltsize);
  unsigned int modulesInThisEvent = 0;
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
    if ( triggerObj->filterIndex(theHLTCollectionLabels[i]) >= 
	 triggerObj->size() )
      LogTrace("HLTMuonVal") << "No HLT Collection with label " << 
	                        theHLTCollectionLabels[i].label();
    else {
      triggerObj->getObjects( 
	triggerObj->filterIndex(theHLTCollectionLabels[i] ), 93, hltCands[i]);
    modulesInThisEvent++;
    }
  }

  if ( useMuonFromGenerator ) theAssociatedGenPart = evt->particles_end(); 
  if ( useMuonFromReco ) theAssociatedRecoPart = muTracks->end(); 

  // Fix L1 thresholds to obtain HLT plots
  unsigned int nL1FoundRef = 0;
  double epsilon = 0.001;
  for ( unsigned int k = 0; k < l1Cands.size(); k++ ) {
    L1MuonParticleRef candRef = L1MuonParticleRef(l1Cands[k]);
    double ptLUT = candRef->pt();

    // Add "epsilon" to avoid rounding errors when ptLUT == L1Threshold
    if ( ptLUT + epsilon > theL1ReferenceThreshold ) {
      nL1FoundRef++;
      hL1pt->Fill(ptLUT);
      if ( useMuonFromGenerator ){
	pair<double,double> angularInfo = 
	  getGenAngle( candRef->eta(), candRef->phi(), *evt );
	LogTrace("HLTMuonVal") << "Filling L1 histos....";
	if ( angularInfo.first < 999.){
	  hL1etaMC->Fill(angularInfo.first);
	  hL1phiMC->Fill(angularInfo.second);
          float dPhi = angularInfo.second - candRef->phi();
          float dEta = angularInfo.first - candRef->eta();
          hL1DR->Fill(sqrt(dPhi*dPhi+dEta*dEta));
	  LogTrace("HLTMuonVal") << "Filling done";
	}
      }
      if ( useMuonFromReco ){
	pair<double,double> angularInfo = getRecAngle( candRef->eta(),
                            candRef->phi(), *muTracks );
	LogTrace("HLTMuonVal") << "Filling L1 histos....";
	if ( angularInfo.first < 999. ){
	  hL1etaRECO->Fill(angularInfo.first);
	  hL1phiRECO->Fill(angularInfo.second);
	  LogTrace("HLTMuonVal") << "Filling done";
	}
      }
    }
  }
  if ( nL1FoundRef >= theNumberOfObjects ){
    if ( genMuonPt > 0 ) hL1MCeff->Fill(genMuonPt, thisEventWeight);
    if ( recMuonPt > 0 ) hL1RECOeff->Fill(recMuonPt, thisEventWeight);
    hSteps->Fill(1.);  
  }

  if ( genMuonPt > 0 ){
    int last_module = modulesInThisEvent - 1;
    for ( int moduleNum = 0; moduleNum <= last_module; moduleNum++) {
      double ptCut = theHLTReferenceThreshold;
      unsigned nFound = 0;
      for ( unsigned int candNum = 0; candNum < hltCands[moduleNum].size(); 
	    candNum++ ) {
	RecoChargedCandidateRef candRef = 
                   RecoChargedCandidateRef(hltCands[moduleNum][candNum]);
	TrackRef track = candRef->get<TrackRef>();
	double pt = track->pt();
	if ( pt > ptCut ) nFound++;
      }
      if ( nFound >= theNumberOfObjects ){
	if ( genMuonPt > 0 ) hHLTMCeff[moduleNum]->Fill( 
						   genMuonPt, thisEventWeight );
	if ( recMuonPt > 0 ) hHLTRECOeff[moduleNum]->Fill( 
						   recMuonPt, thisEventWeight );
	hSteps->Fill( 2 + moduleNum ); 
      }
    }
  }

  for ( unsigned int j = 0; j < theNbins; j++ ) {
    double ptCut = thePtMin + j * (thePtMax - thePtMin) / theNbins;

    // L1 filling
    unsigned int nFound = 0;
    for ( unsigned int candNum = 0; candNum < l1Cands.size(); candNum++) {
      L1MuonParticleRef candRef = L1MuonParticleRef(l1Cands[candNum]);
      double pt = candRef->pt();
      if ( pt > ptCut ) nFound++;
    }
    if ( nFound >= theNumberOfObjects ) hL1eff->Fill(ptCut, thisEventWeight);

    // Stop here if L1 reference cuts were not satisfied
    if ( nL1FoundRef < theNumberOfObjects ) continue;

    // HLT filling
    for ( unsigned int i = 0; i < modulesInThisEvent; i++ ) {
      unsigned nFound = 0;
      for ( unsigned int candNum = 0; candNum < hltCands[i].size(); candNum++) {
	RecoChargedCandidateRef candRef = 
	                        RecoChargedCandidateRef(hltCands[i][candNum]);
	TrackRef track = candRef->get<TrackRef>();
	double pt = track->pt();
	if ( ptCut == thePtMin ) {
	  hHLTpt[i]->Fill(pt);
	  if ( useMuonFromGenerator ){
	    pair<double,double> angularInfo = 
	                getGenAngle( candRef->eta(), candRef->phi(), *evt );
	    if ( angularInfo.first < 999. ) {
	      LogTrace("HLTMuonVal") << "Filling HLT histos for MC [" << 
		                        i << "]........";
	      hHLTetaMC[i]->Fill(angularInfo.first);
	      hHLTphiMC[i]->Fill(angularInfo.second);
	      float dPhi = angularInfo.second - candRef->phi();
	      float dEta = angularInfo.first  - candRef->eta();
	      float dR = sqrt( dPhi*dPhi + dEta*dEta );
	      if ( i == 0 ) hL2DR->Fill(dR);
	      if ( (modulesInThisEvent == 2 && i == 1) || i == 2 ) 
		hL3DR->Fill(dR);
	      LogTrace("HLTMuonVal") << "Filling done";
	    }
	  }
	  if (useMuonFromReco){
	    pair<double, double> angularInfo = getRecAngle( candRef->eta(), 
							     candRef->phi(), 
							     *muTracks );
	    if ( angularInfo.first < 999. ){
	      LogTrace("HLTMuonVal") << "Filling HLT histos for RECO....[" << 
		                        i << "]........";
	      hHLTetaRECO[i]->Fill(angularInfo.first);
	      hHLTphiRECO[i]->Fill(angularInfo.second);
	      LogTrace("HLTMuonVal") << "Filling done";
	    }
	  }
	}
	double err0 = track->error(0);
	double abspar0 = fabs( track->parameter(0) );
	// convert to 90% efficiency threshold
	if ( abspar0 > 0 ) pt += theNSigmas[i] * err0 / abspar0 * pt;
	if ( pt > ptCut ) nFound++;
      }
      if ( nFound >= theNumberOfObjects )
	hHLTeff[i]->Fill(ptCut, thisEventWeight);
    }
  }

}



pair<double,double> HLTMuonGenericRate::getGenAngle( double eta, double phi, 
                    HepMC::GenEvent evt, double candDeltaR )
{
  LogTrace("HLTMuonVal")<< "in getGenAngle";
  HepMC::GenEvent::particle_const_iterator part;
  HepMC::GenEvent::particle_const_iterator theAssociatedpart = 
                                           evt.particles_end();
  pair<double,double> angle( 999., 999. );
  LogTrace("HLTMuonVal") << " candidate eta = " << eta << " and phi = "<< phi;
  for (part = evt.particles_begin(); part != evt.particles_end(); ++part ) {
    int id = abs((*part)->pdg_id());
    if ( id == 13 && (*part)->status() == 1 ) {
      double dEta = eta - (*part)->momentum().eta();
      double dPhi = phi-(*part)->momentum().phi();
      double deltaR = sqrt( dEta * dEta + dPhi * dPhi );
      if ( deltaR < candDeltaR ) {
	candDeltaR = deltaR;
	theAssociatedpart = part;
        angle.first  = (*part)->momentum().eta();
        angle.second = (*part)->momentum().phi();
      }
    }
  }
  LogTrace("HLTMuonVal") << "Best deltaR = " << candDeltaR;
  return angle;
}



pair<double,double> HLTMuonGenericRate::getRecAngle( double eta, double phi, 
                    reco::TrackCollection muTracks,  double candDeltaR )
{
  LogTrace("HLTMuonVal") << "in getRecAngle";
  reco::TrackCollection::const_iterator muon;
  reco::TrackCollection::const_iterator theAssociatedpart = muTracks.end();
  pair<double,double> angle( 999., 999. );
  LogTrace("HLTMuonVal") << " candidate eta = " << eta << " and phi = " << phi;
  for ( muon = muTracks.begin(); muon != muTracks.end(); ++muon ) {
    double dEta = eta - muon->eta();
    double dPhi = phi - muon->phi();
    double deltaR = sqrt( dEta * dEta + dPhi * dPhi );
    if ( deltaR < candDeltaR ) {
      candDeltaR = deltaR;
      theAssociatedpart = muon;
      angle.first  = muon->eta();
      angle.second = muon->phi();
    }
  }
  LogTrace("HLTMuonVal") << "Best deltaR = " << candDeltaR;
  return angle;
}



void HLTMuonGenericRate::BookHistograms(){
  TString dirLabel, myLabel, newFolder, histName, histTitle;
  vector<TH1F*> h;
  if (dbe_) {
    dbe_->cd();
    dbe_->setCurrentFolder("HLT/Muon");

    TString theL1CollectionString( theL1CollectionLabel.encode().c_str() );

    dirLabel = theL1CollectionString;
    dirLabel.Resize( dirLabel.Index("L1") ); // Truncate starting at "L1"
    myLabel = theL1CollectionString;
    myLabel.Resize( myLabel.Index(":") );

    newFolder = "HLT/Muon/RateEfficiencies/" + dirLabel;
    dbe_->setCurrentFolder( newFolder.Data() );

    NumberOfEvents   = dbe_->bookInt("NumberOfEvents");
    NumberOfL1Events = dbe_->bookInt("NumberOfL1Events");

    hL1DR   = BookIt( "L1DR_" + myLabel, "L1 Gen association #DR, label = " + myLabel,  theNbins, 0., 0.5);
    hL2DR   = BookIt( "L2DR_" + myLabel, "L2 Gen association #DR, label=" + myLabel, theNbins, 0., 0.5);
    hL3DR   = BookIt( "L3DR_" + myLabel, "L3 Gen association #DR, label=" + myLabel, theNbins, 0., 0.5);
    hSteps  = BookIt( "HLTSteps_" + myLabel, "Events passing the HLT filters, label=" + myLabel, 5, 0.5, 5.5);
    hL1eff  = BookIt( "eff_" + myLabel, "Efficiency (%%) vs L1 Pt threshold (GeV), label=" + myLabel,  theNbins, thePtMin, thePtMax);
    hL1rate = BookIt( "rate_" + myLabel, "Rate (Hz) vs L1 Pt threshold (GeV), label=" + myLabel,  theNbins, thePtMin, thePtMax);

    dbe_->cd();
    newFolder = "HLT/Muon/Distributions/" + dirLabel;
    dbe_->setCurrentFolder( newFolder.Data() );
    hL1pt = BookIt( "pt_" + myLabel, "L1 Pt distribution label=" + myLabel,  theNbins, thePtMin, thePtMax);
    if (useMuonFromGenerator){
      hL1MCeff  = BookIt( "MCTurnOn_" + myLabel, "L1 max ref pt efficiency label=" + myLabel,  theNbins, thePtMin, thePtMax);
      hMCptnor  = BookIt( "MCptNorm_" + myLabel, "L1 max ref pt distribution label=" + myLabel,  theNbins, thePtMin, thePtMax);
      hMCetanor = BookIt( "MCetaNorm_" + myLabel, "Norm  MC Eta ",  50, -2.1, 2.1);
      hMCphinor = BookIt( "MCphiNorm_" + myLabel, "Norm  MC #Phi",  50, -3.15, 3.15);
      hL1etaMC  = BookIt( "MCeta_" + myLabel, "L1 Eta distribution label=" + myLabel,  50, -2.1, 2.1);
      hL1phiMC  = BookIt( "MCphi_" + myLabel, "L1 Phi distribution label=" + myLabel,  50, -3.15, 3.15);
    }
    if (useMuonFromReco){
      hL1RECOeff  = BookIt( "RECOTurnOn_" + myLabel, "L1 max ref pt efficiency label=" + myLabel,  theNbins, thePtMin, thePtMax);
      hRECOptnor  = BookIt( "RECOptNorm_" + myLabel, "L1 max reco ref pt distribution label=" + myLabel,  theNbins, thePtMin, thePtMax);
      hRECOetanor = BookIt( "RECOetaNorm_" + myLabel, "Norm  RECO Eta ",  50, -2.1, 2.1);
      hRECOphinor = BookIt( "RECOphiNorm_" + myLabel, "Norm  RECO #Phi",  50, -3.15, 3.15);
      hL1etaRECO  = BookIt( "RECOeta_" + myLabel, "L1 Eta distribution label=" + myLabel,  50, -2.1, 2.1);
      hL1phiRECO  = BookIt( "RECOphi_" + myLabel, "L1 Phi distribution label=" + myLabel,  50, -3.15, 3.15);
    }

    for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
      dbe_->cd();
      newFolder = "HLT/Muon/RateEfficiencies/" + dirLabel;
      dbe_->setCurrentFolder( newFolder.Data() );

      TString theHLTCollectionString = theHLTCollectionLabels[i].encode().c_str();
      myLabel = theHLTCollectionString;
      myLabel.Resize( myLabel.Index(":") );

      hHLTeff.push_back(BookIt( "eff_" + myLabel, "Efficiency (%%) vs HLT Pt threshold (GeV), label=" + myLabel,  theNbins, thePtMin, thePtMax));
      hHLTrate.push_back(BookIt( "rate_" + myLabel, "Rate (Hz) vs HLT Pt threshold (GeV), label=" + myLabel,  theNbins, thePtMin, thePtMax));

      dbe_->cd();
      newFolder = "HLT/Muon/Distributions/" + dirLabel;
      dbe_->setCurrentFolder( newFolder.Data() );
      hHLTpt.push_back(BookIt( "pt_" + myLabel, "Pt distribution label=" + myLabel,  theNbins, thePtMin, thePtMax));
      if (useMuonFromGenerator){
	// histTitle = "Turn On curve, label=" + myLabel + ",  L=" + theLuminosity*1.e33 + " (cm^{-2} s^{-1})");
	hHLTMCeff.push_back(BookIt( "MCTurnOn_" + myLabel, "Turn On curve, label=" + myLabel, theNbins, thePtMin, thePtMax));   
	hHLTetaMC.push_back(BookIt( "MCeta_" + myLabel, "Gen Eta Efficiency label=" + myLabel,  50, -2.1, 2.1));
	hHLTphiMC.push_back(BookIt( "MCphi_" + myLabel, "Gen Phi Efficiency label=" + myLabel,  50, -3.15, 3.15));
      }
      if (useMuonFromReco){
	// histTitle = "Turn On curve, label=, L=%.2E (cm^{-2} s^{-1})", myLabel, theLuminosity*1.e33);
	histName = "RECOTurnOn_" + myLabel;
	hHLTRECOeff.push_back(BookIt( "RECOTurnOn_" + myLabel, "Turn On curve, label=" + myLabel, theNbins, thePtMin, thePtMax));     
	hHLTetaRECO.push_back(BookIt( "RECOeta_" + myLabel, "Reco Eta Efficiency label=" + myLabel,  50, -2.1, 2.1));
	hHLTphiRECO.push_back(BookIt( "RECOphi_" + myLabel, "Reco Phi Efficiency label=" + myLabel,  50, -3.15, 3.15));
      }
    }

    hSteps->setAxisTitle("Trigger Filtering Step");
    hSteps->setAxisTitle("Events passing Trigger Step",2);
    hL1eff->setAxisTitle("90% Muon Pt threshold (GeV)");
    hL1rate->setAxisTitle("90% Muon Pt threshold (GeV)");
    hL1rate->setAxisTitle("Rate (Hz)",2);
    if (useMuonFromGenerator){ 
      hL1MCeff->setAxisTitle("Generated Muon PtMax (GeV)");
      hL1MCeff->setAxisTitle("Events Passing L1",2);
    }
    if (useMuonFromReco){
      hL1RECOeff->setAxisTitle("Reconstructed Muon PtMax (GeV)");
      hL1RECOeff->setAxisTitle("Events Passing L1",2);
    }

    hL1pt->setAxisTitle("Muon Pt (GeV)");
    if (useMuonFromGenerator){
      hL1etaMC->setAxisTitle("Muon #eta");
      hL1etaMC->setAxisTitle("Events Passing L1",2);
      hL1phiMC->setAxisTitle("Muon #phi");
      hL1phiMC->setAxisTitle("Events Passing L1",2);
      hMCetanor->setAxisTitle("Gen Muon #eta");
      hMCphinor->setAxisTitle("Gen Muon #phi ");
    }
    if (useMuonFromReco){
      hL1etaRECO->setAxisTitle("Muon #eta");
      hL1etaRECO->setAxisTitle("Events Passing L1",2);
      hL1phiRECO->setAxisTitle("Muon #phi");
      hL1phiRECO->setAxisTitle("Events Passing L1",2);
      hRECOetanor->setAxisTitle("Reco Muon #eta");
      hRECOphinor->setAxisTitle("Reco Muon #phi ");
    }

    for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
      hHLTeff[i]->setAxisTitle("90% Muon Pt threshold (GeV)");
      hHLTrate[i]->setAxisTitle("Rate (Hz)",2);
      hHLTrate[i]->setAxisTitle("90% Muon Pt threshold (GeV)",1);
      hHLTpt[i]->setAxisTitle("HLT Muon Pt (GeV)",1);
      if (useMuonFromGenerator){
	hHLTMCeff[i]->setAxisTitle("Generated Muon PtMax (GeV)",1);
	hHLTMCeff[i]->setAxisTitle("Events Passing Trigger",2);
	hHLTetaMC[i]->setAxisTitle("Gen Muon #eta",1);
	hHLTetaMC[i]->setAxisTitle("Events Passing Trigger",2);
	hHLTphiMC[i]->setAxisTitle("Gen Muon #phi",1);
	hHLTphiMC[i]->setAxisTitle("Events Passing Trigger",2);
      }
      if (useMuonFromReco){
	hHLTRECOeff[i]->setAxisTitle("Reconstructed Muon PtMax (GeV)",1);
	hHLTRECOeff[i]->setAxisTitle("Events Passing Trigger",2);
	hHLTetaRECO[i]->setAxisTitle("Reco Muon #eta",1);	
	hHLTetaRECO[i]->setAxisTitle("Events Passing Trigger",2);
	hHLTphiRECO[i]->setAxisTitle("Reco Muon #phi",1);
	hHLTphiRECO[i]->setAxisTitle("Events Passing Trigger",2);
      }
    }
  }
}



MonitorElement* HLTMuonGenericRate::BookIt( TString name, TString title, 
					    int Nbins, float Min, float Max) {
  LogDebug("HLTMuonVal") << "Directory " << dbe_->pwd() << " Name " << 
                            name << " Title:" << title;
  TH1F *h = new TH1F( name, title, Nbins, Min, Max );
  h->Sumw2();
  return dbe_->book1D( name.Data(), h );
  delete h;
}



void HLTMuonGenericRate::WriteHistograms() {
  if ( theRootFileName.size() != 0 && dbe_ ) dbe_->save(theRootFileName);
   return;
}


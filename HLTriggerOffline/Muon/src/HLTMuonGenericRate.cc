 /** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author M. Vander Donckt, J. Klukas  (copied from J. Alcaraz)
 */

#include "HLTriggerOffline/Muon/interface/HLTMuonGenericRate.h"
#include "HLTriggerOffline/Muon/interface/AnglesUtil.h"

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
HLTMuonGenericRate::HLTMuonGenericRate( const ParameterSet& pset, 
					int triggerIndex )
{
  useMuonFromGenerator = pset.getParameter<bool>("UseMuonFromGenerator");
  useMuonFromReco = pset.getParameter<bool>("UseMuonFromReco");
  if ( useMuonFromGenerator ) 
    theGenLabel = pset.getUntrackedParameter<InputTag>("GenLabel");
  if ( useMuonFromReco )
    theRecoLabel = pset.getUntrackedParameter<InputTag>("RecoLabel");
  
  folderName = pset.getUntrackedParameter<string>("FolderName", "HLT/Muon/");

  Parameters TriggerLists  = pset.getParameter<Parameters>("TriggerCollection");
  ParameterSet thisTrigger = TriggerLists[triggerIndex];
  theL1CollectionLabel     = thisTrigger.getParameter<InputTag>("L1CollectionLabel");
  theHLTCollectionLabels   = thisTrigger.getParameter< std::vector<InputTag> >("HLTCollectionLabels");
  theL1ReferenceThreshold  = thisTrigger.getParameter<double>("L1ReferenceThreshold");    
  theHLTReferenceThreshold = thisTrigger.getParameter<double>("HLTReferenceThreshold");    
  theNumberOfObjects       = thisTrigger.getParameter<unsigned int>("NumberOfObjects");

  theNSigmas      = pset.getUntrackedParameter<std::vector<double> >("NSigmas90");
  theCrossSection = pset.getParameter<double>("CrossSection");

  // Luminosity converted to nb^-1 * s^-1
  theLuminosity = pset.getUntrackedParameter<double>("Luminosity",1.e+32)*1.e-33;

  thePtMin = pset.getUntrackedParameter<double>("PtMin",0.);
  thePtMax = pset.getUntrackedParameter<double>("PtMax",40.);
  theNbins = pset.getUntrackedParameter<unsigned int>("Nbins",40);
  
  theNumberOfEvents   = 0;
  theNumberOfL1Events = 0;

  dbe_ = 0 ;
  if ( pset.getUntrackedParameter<bool>("DQMStore", false) ) {
    dbe_ = Service<DQMStore>().operator->();
    dbe_->setVerbose(0);
  }

  if ( pset.getUntrackedParameter<bool>("disableROOToutput", false) ) 
    theRootFileName="";
  else 
    theRootFileName = pset.getUntrackedParameter<std::string>("RootFileName");

  if ( dbe_ != NULL ) {
    dbe_->cd();
    dbe_->setCurrentFolder( folderName );
    dbe_->setCurrentFolder( std::string( folderName + "RateEfficiencies" ) );
    dbe_->setCurrentFolder( std::string( folderName + "Distributions" ) );
  }
}



/// Destructor
HLTMuonGenericRate::~HLTMuonGenericRate()
{
}



void HLTMuonGenericRate::analyze( const Event & event )
{
  thisEventWeight = 1;
  NumberOfEvents->Fill(++theNumberOfEvents); // Sets ME<int> NumberOfEvents
  LogTrace( "HLTMuonVal" ) << "In analyze for L1 trigger " << 
    theL1CollectionLabel << " Event:" << theNumberOfEvents;  


  //////////////////////////////////////////////////////////////////////////
  // Get all generated and reconstructed muons, fill eta and phi histograms,
  // and save the highest pt from each collection

  bool foundGenMuon = false;
  bool foundRecMuon  = false;

  double genMuonPt = -1;
  double recMuonPt = -1;

  if (useMuonFromGenerator) {
    Handle<HepMCProduct> genProduct;
    event.getByLabel(theGenLabel, genProduct);
    if ( genProduct.failedToGet() ){
      LogDebug("HLTMuonVal") << "No generator input to compare to";
      useMuonFromGenerator = false;
    } else {
      evt = genProduct->GetEvent();
      HepMC::GenEvent::particle_const_iterator genIterator;
      for ( genIterator = evt->particles_begin(); 
	    genIterator != evt->particles_end(); ++genIterator ) {
	int id     = (*genIterator)->pdg_id();
	int status = (*genIterator)->status();
	if ( abs(id) == 13 && status == 1 ) {
	  float pt = (*genIterator)->momentum().perp();
	  hMCetanor->Fill((*genIterator)->momentum().eta());
	  hMCphinor->Fill((*genIterator)->momentum().phi());
	  if ( pt > genMuonPt) {
	    foundGenMuon = true;
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
      LogDebug("HLTMuonVal") << "No reco tracks to compare to";
      useMuonFromReco = false;
    } else {
      for ( muon = muTracks->begin(); muon != muTracks->end(); ++muon ) {
	float pt = muon->pt();
	hRECOetanor->Fill( muon->eta() );
	hRECOphinor->Fill( muon->phi() );
	if ( pt > recMuonPt ) {
	  foundRecMuon  = true;
	  recMuonPt = pt;
	}
      }
    }
  } 
  
  if ( foundGenMuon ) hMCMaxPt->Fill(genMuonPt, thisEventWeight);
  if ( foundRecMuon ) hRECOMaxPt->Fill(recMuonPt, thisEventWeight);

  LogTrace("HLTMuonVal") << "genMuonPt: " << genMuonPt
                         << "recMuonPt: " << recMuonPt;

  //////////////////////////////////////////////////////////////////////////
  // Get the L1 and HLT trigger collections

  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByLabel("hltTriggerSummaryRAW", triggerObj); 

  if( !triggerObj.isValid() ) { 
    LogDebug("HLTMuonVal") << "RAW-type HLT results not found, skipping event";
    return;
  }

  // Output filter information
  const size_type numFilterObjects(triggerObj->size());
  LogTrace("HLTMuonVal") 
    << "Used Processname: " << triggerObj->usedProcessName() << "\n"
    << "Number of TriggerFilterObjects: " << numFilterObjects << "\n"
    << "The TriggerFilterObjects: #, tag";
  for ( size_type i = 0; i != numFilterObjects; ++i ) LogTrace("HLTMuonVal") 
    << i << " " << triggerObj->filterTag(i).encode();

  InputTag label; // Used as a shortcut for the current CollectionLabel

  // Get the L1 candidates //
  vector<L1MuonParticleRef> l1Cands;
  label = theL1CollectionLabel;
  LogTrace("HLTMuonVal") << "TriggerObject Size=" << triggerObj->size();
  if ( triggerObj->filterIndex(label) >= triggerObj->size() ) {
    LogTrace("HLTMuonVal") << "No L1 Collection with label " << label;
    return;
  }
  triggerObj->getObjects( triggerObj->filterIndex(label), 81, l1Cands );
  NumberOfL1Events->Fill( ++theNumberOfL1Events );

  // Get the HLT candidates //
  unsigned int numHltLabels = theHLTCollectionLabels.size();
  vector< vector<RecoChargedCandidateRef> > hltCands(numHltLabels);
  unsigned int numHltModules = 0;
  for ( unsigned int i = 0; i < numHltLabels; i++ ) {
    label = theHLTCollectionLabels[i];
    if ( triggerObj->filterIndex(label) >= triggerObj->size() )
      LogTrace("HLTMuonVal") << "No HLT Collection with label " << label.label();
    else {
      triggerObj->getObjects( triggerObj->filterIndex(label), 93, hltCands[i] );
      numHltModules++;
    }
  }


  //////////////////////////////////////////////////////////////////////////
  // Find L1 candidates passing a pt threshold, fill histograms, and match
  // them to generated and reconstructed muons

  unsigned int nL1FoundRef = 0;

  for ( size_t i = 0; i < l1Cands.size(); i++ ) {

    L1MuonParticleRef l1Cand = L1MuonParticleRef( l1Cands[i] );
    double eta = l1Cand->eta();
    double phi = l1Cand->phi();
    double ptLUT = l1Cand->pt(); // L1 pt is taken from a lookup table
    double pt = ptLUT + 0.001;   // In case ptLUT, which is discrete, exactly
                                 // equals theL1ReferenceThreshold

    if ( pt > theL1ReferenceThreshold ) {
      double maxDeltaR = 0.4;
      nL1FoundRef++;
      hL1pt->Fill(pt);
      if ( useMuonFromGenerator ){
	pair<double,double> angles = getAngles( eta, phi, *evt, maxDeltaR );
	if ( angles.first < 999.){
	  hL1etaMC->Fill(angles.first);
	  hL1phiMC->Fill(angles.second);
          float dPhi = angles.second - l1Cand->phi();
          float dEta = angles.first - l1Cand->eta();
          hL1DR->Fill( sqrt(dPhi*dPhi + dEta*dEta) );
	}
      }
      if ( useMuonFromReco ){
	pair<double,double> angles = getAngles( eta, phi, *muTracks, maxDeltaR );
	if ( angles.first < 999. ){
	  hL1etaRECO->Fill(angles.first);
	  hL1phiRECO->Fill(angles.second);
	}
      }
    }

  }

  if ( nL1FoundRef >= theNumberOfObjects ){
    if ( foundGenMuon ) hMCMaxPtPassL1->Fill(genMuonPt, thisEventWeight);
    if ( foundRecMuon ) hRECOMaxPtPassL1->Fill(recMuonPt, thisEventWeight);
    hSteps->Fill(1.);  
  }


  //////////////////////////////////////////////////////////////////////////
  // Count the number of candidates for each HLT module that pass a pt 
  // threshold, and fill efficiency numerator histograms

  if ( foundGenMuon ){
    for ( size_t moduleNum = 0; moduleNum < numHltModules; moduleNum++) {

      double ptCut = theHLTReferenceThreshold;
      unsigned int nFound = 0;

      for ( size_t candNum = 0; candNum < hltCands[moduleNum].size(); candNum++ ) {
	RecoChargedCandidateRef hltCand = hltCands[moduleNum][candNum];
	TrackRef track = hltCand->get<TrackRef>();
	if ( track->pt() > ptCut ) nFound++;
      }

      if ( nFound >= theNumberOfObjects ){
	if ( foundGenMuon ) hHLTMCMaxPtPass[moduleNum]->Fill( genMuonPt, thisEventWeight );
	if ( foundRecMuon ) hHLTRECOMaxPtPass[moduleNum]->Fill( recMuonPt, thisEventWeight );
	hSteps->Fill( 2 + moduleNum ); 
      }

    }
  }


  //////////////////////////////////////////////////////////////////////////
  // Find HLT candidates for each module passing a pt threshold, fill
  // histograms, and match them to generated and reconstructed muons

  for ( size_t  moduleNum = 0; moduleNum < numHltModules; moduleNum++ ) { 
    for ( size_t candNum = 0; candNum < hltCands[moduleNum].size(); candNum++ ) {

      int triggerLevel;
      double maxDeltaR;

      if ( moduleNum < ( numHltModules / 2 ) ) {
	triggerLevel = 2;
	maxDeltaR = 0.05;
      } else {
	triggerLevel = 3;
	maxDeltaR = 0.015;
      }

      RecoChargedCandidateRef hltCand = hltCands[moduleNum][candNum];
      double eta = hltCand->eta();
      double phi = hltCand->phi();
      double pt  = hltCand->pt();
      hHLTpt[moduleNum]->Fill(pt);

      if ( useMuonFromGenerator ){
	pair<double,double> angles = getAngles( eta, phi, *evt, maxDeltaR );
	if ( angles.first < 999. ) {
	  hHLTetaMC[moduleNum]->Fill(angles.first);
	  hHLTphiMC[moduleNum]->Fill(angles.second);
	  float dPhi = angles.second - phi;
	  float dEta = angles.first  - eta;
	  float deltaR = sqrt( dPhi*dPhi + dEta*dEta );
	  if ( triggerLevel == 2 ) hL2DR->Fill(deltaR);
	  else                     hL3DR->Fill(deltaR);
	}
      }

      if ( useMuonFromReco ){
	pair<double, double> angles = getAngles( eta, phi, *muTracks, maxDeltaR );
	if ( angles.first < 999. ){
	  hHLTetaRECO[moduleNum]->Fill(angles.first);
	  hHLTphiRECO[moduleNum]->Fill(angles.second);
	}
      }

    }
  }


  //////////////////////////////////////////////////////////////////////////
  // Cycle through various pt cuts and fill efficiency histograms using
  // 90% confidence level estimates of pt

  for ( unsigned int binNum = 0; binNum < theNbins; binNum++ ) {
    double ptCut = thePtMin + binNum * (thePtMax - thePtMin) / theNbins;

    // L1 filling
    unsigned int nFound = 0;
    for ( size_t candNum = 0; candNum < l1Cands.size(); candNum++) {
      L1MuonParticleRef l1Cand = l1Cands[candNum];
      double pt = l1Cand->pt();
      if ( pt > ptCut ) nFound++;
    }
    if ( nFound >= theNumberOfObjects ) hL1eff->Fill(ptCut, thisEventWeight);
    if ( nL1FoundRef < theNumberOfObjects ) continue; // CHECK THIS!!

    // HLT filling
    for ( size_t i = 0; i < numHltModules; i++ ) {
      unsigned int nFound = 0;
      for ( size_t candNum = 0; candNum < hltCands[i].size(); candNum++) {
	RecoChargedCandidateRef hltCand =  hltCands[i][candNum];
	TrackRef track = hltCand->get<TrackRef>();
	double pt = track->pt();
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



pair<double,double> HLTMuonGenericRate::getAngles( double eta, double phi, 
                    HepMC::GenEvent evt, double maxDeltaR )
{
  HepMC::GenEvent::particle_const_iterator part;
  HepMC::GenEvent::particle_const_iterator theAssociatedGenParticle = 
                                           evt.particles_end();
  pair<double,double> angle( 999., 999. );
  double bestDeltaR = maxDeltaR;
  LogTrace("HLTMuonVal") << " candidate eta = " << eta << " and phi = "<< phi;

  for (part = evt.particles_begin(); part != evt.particles_end(); ++part ) {
    int id = abs((*part)->pdg_id());
    if ( id == 13 && (*part)->status() == 1 ) {

      double genEta = (*part)->momentum().eta();
      double genPhi = (*part)->momentum().phi();
      double deltaR = kinem::delta_R( eta, phi, genEta, genPhi );
      if ( deltaR < bestDeltaR ) {
	bestDeltaR = deltaR;
	theAssociatedGenParticle = part;
        angle.first  = genEta;
        angle.second = genPhi;
      }
    }

  }
  LogTrace("HLTMuonVal") << "Best deltaR = " << bestDeltaR;
  return angle;
}



pair<double,double> HLTMuonGenericRate::getAngles( double eta, double phi, 
                    reco::TrackCollection muTracks,  double maxDeltaR )
{
  reco::TrackCollection::const_iterator muon;
  reco::TrackCollection::const_iterator theAssociatedRecParticle = 
                                        muTracks.end();
  pair<double,double> angle( 999., 999. );
  double bestDeltaR = maxDeltaR;
  LogTrace("HLTMuonVal") << " candidate eta = " << eta << " and phi = " << phi;

  for ( muon = muTracks.begin(); muon != muTracks.end(); ++muon ) {

    double deltaR = kinem::delta_R( eta, phi, muon->eta(), muon->phi() );
    if ( deltaR < bestDeltaR ) {
      bestDeltaR = deltaR;
      theAssociatedRecParticle = muon;
      angle.first  = muon->eta();
      angle.second = muon->phi();
    }

  }
  LogTrace("HLTMuonVal") << "Best deltaR = " << bestDeltaR;
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
      hMCMaxPtPassL1  = BookIt( "MCMaxPtPass_" + myLabel, "L1 max ref pt efficiency label=" + myLabel,  theNbins, thePtMin, thePtMax);
      hMCMaxPt  = BookIt( "MCMaxPt_" + myLabel, "L1 max ref pt distribution label=" + myLabel,  theNbins, thePtMin, thePtMax);
      hMCetanor = BookIt( "MCetaNorm_" + myLabel, "Norm  MC Eta ",  50, -2.1, 2.1);
      hMCphinor = BookIt( "MCphiNorm_" + myLabel, "Norm  MC #Phi",  50, -3.15, 3.15);
      hL1etaMC  = BookIt( "MCNumeta_" + myLabel, "L1 Eta distribution label=" + myLabel,  50, -2.1, 2.1);
      hL1phiMC  = BookIt( "MCNumphi_" + myLabel, "L1 Phi distribution label=" + myLabel,  50, -3.15, 3.15);
    }
    if (useMuonFromReco){
      hRECOMaxPtPassL1  = BookIt( "RECOMaxPtPass_" + myLabel, "L1 max ref pt efficiency label=" + myLabel,  theNbins, thePtMin, thePtMax);
      hRECOMaxPt  = BookIt( "RECOMaxPt_" + myLabel, "L1 max reco ref pt distribution label=" + myLabel,  theNbins, thePtMin, thePtMax);
      hRECOetanor = BookIt( "RECOetaNorm_" + myLabel, "Norm  RECO Eta ",  50, -2.1, 2.1);
      hRECOphinor = BookIt( "RECOphiNorm_" + myLabel, "Norm  RECO #Phi",  50, -3.15, 3.15);
      hL1etaRECO  = BookIt( "RECONumeta_" + myLabel, "L1 Eta distribution label=" + myLabel,  50, -2.1, 2.1);
      hL1phiRECO  = BookIt( "RECONumphi_" + myLabel, "L1 Phi distribution label=" + myLabel,  50, -3.15, 3.15);
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
	hHLTMCMaxPtPass.push_back(BookIt( "MCMaxPtPass_" + myLabel, "Turn On curve, label=" + myLabel, theNbins, thePtMin, thePtMax));   
	hHLTetaMC.push_back(BookIt( "MCNumeta_" + myLabel, "Gen Eta Efficiency label=" + myLabel,  50, -2.1, 2.1));
	hHLTphiMC.push_back(BookIt( "MCNumphi_" + myLabel, "Gen Phi Efficiency label=" + myLabel,  50, -3.15, 3.15));
      }
      if (useMuonFromReco){
	// histTitle = "Turn On curve, label=, L=%.2E (cm^{-2} s^{-1})", myLabel, theLuminosity*1.e33);
	histName = "RECOTurnOn_" + myLabel;
	hHLTRECOMaxPtPass.push_back(BookIt( "RECOMaxPtPass_" + myLabel, "Turn On curve, label=" + myLabel, theNbins, thePtMin, thePtMax));     
	hHLTetaRECO.push_back(BookIt( "RECONumeta_" + myLabel, "Reco Eta Efficiency label=" + myLabel,  50, -2.1, 2.1));
	hHLTphiRECO.push_back(BookIt( "RECONumphi_" + myLabel, "Reco Phi Efficiency label=" + myLabel,  50, -3.15, 3.15));
      }
    }

    hSteps->setAxisTitle("Trigger Filtering Step");
    hSteps->setAxisTitle("Events passing Trigger Step",2);
    hL1eff->setAxisTitle("90% Muon Pt threshold (GeV)");
    hL1rate->setAxisTitle("90% Muon Pt threshold (GeV)");
    hL1rate->setAxisTitle("Rate (Hz)",2);
    if (useMuonFromGenerator){ 
      hMCMaxPtPassL1->setAxisTitle("Generated Muon p_{T}^{Max} (GeV)");
      hMCMaxPtPassL1->setAxisTitle("Events Passing L1",2);
    }
    if (useMuonFromReco){
      hRECOMaxPtPassL1->setAxisTitle("Reconstructed Muon p_{T}^{Max} (GeV)");
      hRECOMaxPtPassL1->setAxisTitle("Events Passing L1",2);
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
	hHLTMCMaxPtPass[i]->setAxisTitle("Generated Muon PtMax (GeV)",1);
	hHLTMCMaxPtPass[i]->setAxisTitle("Events Passing Trigger",2);
	hHLTetaMC[i]->setAxisTitle("Gen Muon #eta",1);
	hHLTetaMC[i]->setAxisTitle("Events Passing Trigger",2);
	hHLTphiMC[i]->setAxisTitle("Gen Muon #phi",1);
	hHLTphiMC[i]->setAxisTitle("Events Passing Trigger",2);
      }
      if (useMuonFromReco){
	hHLTRECOMaxPtPass[i]->setAxisTitle("Reconstructed Muon PtMax (GeV)",1);
	hHLTRECOMaxPtPass[i]->setAxisTitle("Events Passing Trigger",2);
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


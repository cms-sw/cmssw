/** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author M. Vander Donckt  (coped from J. Alcaraz)
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
HLTMuonGenericRate::HLTMuonGenericRate(const ParameterSet& pset, int Index)
{
  useMuonFromGenerator = pset.getParameter<bool>("UseMuonFromGenerator");
  useMuonFromReco = pset.getParameter<bool>("UseMuonFromReco");
  if(useMuonFromGenerator)theGenLabel = pset.getUntrackedParameter<InputTag>("GenLabel");
  if(useMuonFromReco)theRecoLabel = pset.getUntrackedParameter<InputTag>("RecoLabel");
  Parameters TriggerLists=pset.getParameter<Parameters>("TriggerCollection");
  int i=0;
  for(Parameters::iterator itTrigger = TriggerLists.begin(); itTrigger != TriggerLists.end(); ++itTrigger) {
    if ( i == Index ) { 
      theL1CollectionLabel = itTrigger->getParameter<InputTag>("L1CollectionLabel");
      theHLTCollectionLabels = itTrigger->getParameter<std::vector<InputTag> >("HLTCollectionLabels");
      theL1ReferenceThreshold = itTrigger->getParameter<double>("L1ReferenceThreshold");    
      theHLTReferenceThreshold = itTrigger->getParameter<double>("HLTReferenceThreshold");    
      theNumberOfObjects = itTrigger->getParameter<unsigned int>("NumberOfObjects");
      break;
    }
    ++i;
  }
  theNSigmas = pset.getUntrackedParameter<std::vector<double> >("NSigmas90");
  theCrossSection = pset.getParameter<double>("CrossSection");
  // Convert it already into /nb/s)
  theLuminosity = pset.getUntrackedParameter<double>("Luminosity",1.e+32)*1.e-33;
  thePtMin = pset.getUntrackedParameter<double>("PtMin",0.);
  thePtMax = pset.getUntrackedParameter<double>("PtMax",40.);
  theNbins = pset.getUntrackedParameter<unsigned int>("Nbins",40);
  theNumberOfEvents = 0;
  theNumberOfL1Events = 0;
  dbe_ = 0 ;
  if (pset.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_ = Service < DQMStore > ().operator->();
    dbe_->setVerbose(0);
  }
  bool disable =pset.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) theRootFileName="";
  else theRootFileName = pset.getUntrackedParameter<std::string>("RootFileName");
  if (dbe_ != NULL) {
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
  this_event_weight=1;
  NumberOfEvents->Fill(++theNumberOfEvents);
  LogTrace("HLTMuonVal")<<"In analyze for L1 trigger "<<theL1CollectionLabel<<" Event:"<<theNumberOfEvents;  
  // Get the muon with maximum pt at generator level or reconstruction, depending on the choice
  bool refmuon_found = false;
  bool refrecomuon_found = false;
  double ptuse = -1;
  double recoptuse = -1;
  if (useMuonFromGenerator) {
    Handle<HepMCProduct> genProduct;
    event.getByLabel(theGenLabel,genProduct);
    if (genProduct.failedToGet()){
      LogWarning("HLTMuonVal")<<"No generator input to compare to";
      useMuonFromGenerator=false;
    } else {
      evt= genProduct->GetEvent();
      HepMC::GenEvent::particle_const_iterator part;
      for (part = evt->particles_begin(); part != evt->particles_end(); ++part ) {
	int id1 = (*part)->pdg_id();
	if (abs(id1)==13 && (*part)->status() == 1  ){
	  float pt1 = (*part)->momentum().perp();
	  hMCetanor->Fill((*part)->momentum().eta());
	  hMCphinor->Fill((*part)->momentum().phi());
	  if (pt1>ptuse) {
	    refmuon_found = true;
	    ptuse = pt1;
	  }
	}
      }
    } 
  }
  Handle<reco::TrackCollection> muTracks;
  if (useMuonFromReco) {
    // Get the muon track collection from the event
    reco::TrackCollection::const_iterator muon;
    event.getByLabel(theRecoLabel.label(), muTracks);    
    if  ( muTracks.failedToGet() ) {
      LogWarning("HLTMuonVal")<<"No reco tracks to compare to";
      useMuonFromReco=false;
      return;
    } else {
      for ( muon = muTracks->begin(); muon != muTracks->end(); ++muon ) {
	float pt1 = muon->pt();
	hRECOetanor->Fill(muon->eta());
	hRECOphinor->Fill(muon->phi());
	if (pt1>recoptuse) {
	  refrecomuon_found = true;
	  recoptuse = pt1;
	}
      }
    }
  } 


  
  if (ptuse > 0 ) hMCptnor->Fill(ptuse,this_event_weight);
  if (recoptuse > 0 ) hRECOptnor->Fill(recoptuse,this_event_weight);

  // Get the Trigger collection
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByLabel("hltTriggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogWarning("HLTMuonVal") << "RAW-type HLT results not found, skipping event";
    return;
  } else {
     LogTrace("HLTMuonVal") << "Used Processname: " << triggerObj->usedProcessName() ;
     const size_type nFO(triggerObj->size());
     LogTrace("HLTMuonVal") << "Number of TriggerFilterObjects: " << nFO;
     LogTrace("HLTMuonVal") << "The TriggerFilterObjects: #, tag" ;
     for (size_type iFO=0; iFO!=nFO; ++iFO) {
      LogTrace("HLTMuonVal") << iFO << " " << triggerObj->filterTag(iFO).encode()   ;
     }
  }

  vector<L1MuonParticleRef> l1cands;
  LogTrace("HLTMuonVal")<<"TriggerObject Size="<<triggerObj->size();
  if ( triggerObj->filterIndex(theL1CollectionLabel)>=triggerObj->size() ){
    LogTrace("HLTMuonVal")<<"No L1 Collection with label "<<theL1CollectionLabel;
    return;
  }
  triggerObj->getObjects(triggerObj->filterIndex(theL1CollectionLabel),81,l1cands);
  NumberOfL1Events->Fill(++theNumberOfL1Events);
  // Get the HLT collections
  unsigned hltsize=theHLTCollectionLabels.size();
  vector<vector<RecoChargedCandidateRef> > hltcands(hltsize);
  unsigned int modules_in_this_event = 0;
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
    if ( triggerObj->filterIndex(theHLTCollectionLabels[i])>=triggerObj->size() ) {
      LogTrace("HLTMuonVal")<<"No HLT Collection with label "<<theHLTCollectionLabels[i].label();
      break ;
    }
    triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[i]),93, hltcands[i]);
    modules_in_this_event++;
  }


  if (useMuonFromGenerator) theAssociatedGenPart=evt->particles_end(); 
  if (useMuonFromReco) theAssociatedRecoPart=muTracks->end(); 

  // Fix L1 thresholds to obtain HLT plots
  unsigned int nL1FoundRef = 0;
  double epsilon = 0.001;
  for (unsigned int k=0; k<l1cands.size(); k++) {
    L1MuonParticleRef candref = L1MuonParticleRef(l1cands[k]);
    double ptLUT = candref->pt();
    // Add "epsilon" to avoid rounding errors when ptLUT==L1Threshold
    if (ptLUT+epsilon>theL1ReferenceThreshold) {
      nL1FoundRef++;
      hL1pt->Fill(ptLUT);
      if (useMuonFromGenerator){
	pair<double,double> angularInfo=getGenAngle(candref->eta(),candref->phi(), *evt );
	LogTrace("HLTMuonVal")<<"Filling L1 histos....";
	if ( angularInfo.first < 999.){
	  hL1etaMC->Fill(angularInfo.first);
	  hL1phiMC->Fill(angularInfo.second);
          float dphi=angularInfo.second-candref->phi();
          float deta=angularInfo.first-candref->eta();
          hL1DR->Fill(sqrt(dphi*dphi+deta*deta));
	  LogTrace("HLTMuonVal")<<"Filling done";
	}
      }
      if (useMuonFromReco){
	pair<double,double> angularInfo=getRecoAngle(candref->eta(),candref->phi(), *muTracks );
	LogTrace("HLTMuonVal")<<"Filling L1 histos....";
	if ( angularInfo.first < 999.){
	  hL1etaRECO->Fill(angularInfo.first);
	  hL1phiRECO->Fill(angularInfo.second);
	  LogTrace("HLTMuonVal")<<"Filling done";
	}
      }
    }
  }
  if (nL1FoundRef>=theNumberOfObjects){
    if(ptuse>0) hL1MCeff->Fill(ptuse,this_event_weight);
    if(recoptuse>0) hL1RECOeff->Fill(recoptuse,this_event_weight);
    hSteps->Fill(1.);  
  }

  if (ptuse>0){
    int last_module = modules_in_this_event - 1;
    for ( int i=0; i<=last_module; i++) {
      double ptcut = theHLTReferenceThreshold;
      unsigned nFound = 0;
      for (unsigned int k=0; k<hltcands[i].size(); k++) {
	RecoChargedCandidateRef candref = RecoChargedCandidateRef(hltcands[i][k]);
	TrackRef tk = candref->get<TrackRef>();
	double pt = tk->pt();
	if (pt>ptcut) nFound++;
      }
      if (nFound>=theNumberOfObjects){
	if(ptuse>0) hHLTMCeff[i]->Fill(ptuse,this_event_weight);
	if(recoptuse>0 ) hHLTRECOeff[i]->Fill(recoptuse,this_event_weight);
	hSteps->Fill(2+i); 
      }
    }
  }

  for (unsigned int j=0; j<theNbins; j++) {
    double ptcut = thePtMin + j*(thePtMax-thePtMin)/theNbins;

    // L1 filling
    unsigned int nFound = 0;
    for (unsigned int k=0; k<l1cands.size(); k++) {
      L1MuonParticleRef candref = L1MuonParticleRef(l1cands[k]);
      double pt = candref->pt();
      if (pt>ptcut) nFound++;
    }
    if (nFound>=theNumberOfObjects) hL1eff->Fill(ptcut,this_event_weight);

    // Stop here if L1 reference cuts were not satisfied
    if (nL1FoundRef<theNumberOfObjects) continue;

    // HLT filling
    for (unsigned int i=0; i<modules_in_this_event; i++) {
      unsigned nFound = 0;
      for (unsigned int k=0; k<hltcands[i].size(); k++) {
	RecoChargedCandidateRef candref = RecoChargedCandidateRef(hltcands[i][k]);
	TrackRef tk = candref->get<TrackRef>();
	double pt = tk->pt();
	if ( ptcut == thePtMin ) {
	  hHLTpt[i]->Fill(pt);
	  if (useMuonFromGenerator){
	    pair<double,double> angularInfo=getGenAngle(candref->eta(),candref->phi(), *evt );
	    if ( angularInfo.first < 999.){
	      LogTrace("HLTMuonVal")<<"Filling HLT histos for MC ["<<i<<"]........";
	      hHLTetaMC[i]->Fill(angularInfo.first);
	      hHLTphiMC[i]->Fill(angularInfo.second);
	      float dphi=angularInfo.second-candref->phi();
	      float deta=angularInfo.first-candref->eta();
	      float d=sqrt(dphi*dphi+deta*deta);
	      if (i==0)hL2DR->Fill(d);
	      if ((modules_in_this_event==2 && i==1)||i==2 )hL3DR->Fill(d);
	      LogTrace("HLTMuonVal")<<"Filling done";
	    }
	  }
	  if (useMuonFromReco){
	    pair<double,double> angularInfo=getRecoAngle(candref->eta(),candref->phi(), *muTracks );
	    if ( angularInfo.first < 999.){
	      LogTrace("HLTMuonVal")<<"Filling HLT histos for RECO....["<<i<<"]........";
	      hHLTetaRECO[i]->Fill(angularInfo.first);
	      hHLTphiRECO[i]->Fill(angularInfo.second);
	      LogTrace("HLTMuonVal")<<"Filling done";
	    }
	  }
	}
	double err0 = tk->error(0);
	double abspar0 = fabs(tk->parameter(0));
	// convert to 90% efficiency threshold
	if (abspar0>0) pt += theNSigmas[i]*err0/abspar0*pt;
	if (pt>ptcut) nFound++;
      }
      if (nFound>=theNumberOfObjects) {
	hHLTeff[i]->Fill(ptcut,this_event_weight);
      } else {
	break;
      }
    }
  }

}

pair<double,double> HLTMuonGenericRate::getGenAngle(double eta, double phi, HepMC::GenEvent evt,double candDeltaR   )
{

  LogTrace("HLTMuonVal")<< "in getGenAngle";
  HepMC::GenEvent::particle_const_iterator part;
  HepMC::GenEvent::particle_const_iterator theAssociatedpart=evt.particles_end();
  pair<double,double> angle(999.,999.);
  LogTrace("HLTMuonVal")<< " candidate eta="<<eta<<" and phi="<<phi;
  for (part = evt.particles_begin(); part != evt.particles_end(); ++part ) {
    int id = abs((*part)->pdg_id());
    if ( id == 13 && (*part)->status() == 1 ) {
      double Deta=eta-(*part)->momentum().eta();
      double Dphi=phi-(*part)->momentum().phi();
      double deltaR = sqrt(Deta*Deta+Dphi*Dphi);
      if ( deltaR < candDeltaR ) {
	candDeltaR=deltaR;
	theAssociatedpart=part;
        angle.first=(*part)->momentum().eta();
        angle.second=(*part)->momentum().phi();
      }
    }
  }
  LogTrace("HLTMuonVal")<< "Best deltaR="<<candDeltaR;
  return angle;

}
pair<double,double> HLTMuonGenericRate::getRecoAngle(double eta, double phi, reco::TrackCollection muTracks,  double candDeltaR )
{

  LogTrace("HLTMuonVal")<< "in getRecoAngle";
  reco::TrackCollection::const_iterator muon;
  reco::TrackCollection::const_iterator theAssociatedpart=muTracks.end();
  pair<double,double> angle(999.,999.);
  LogTrace("HLTMuonVal")<< " candidate eta="<<eta<<" and phi="<<phi;
  for (muon = muTracks.begin(); muon != muTracks.end(); ++muon ) {
    double Deta=eta-muon->eta();
    double Dphi=phi-muon->phi();
    double deltaR = sqrt(Deta*Deta+Dphi*Dphi);
    if ( deltaR < candDeltaR ) {
      candDeltaR=deltaR;
      theAssociatedpart=muon;
      angle.first=muon->eta();
      angle.second=muon->phi();
    }
  }
  LogTrace("HLTMuonVal")<< "Best deltaR="<<candDeltaR;
  return angle;

}

void HLTMuonGenericRate::BookHistograms(){
  char chname[256];
  char chtitle[256];
  char * mylabel ;
  char * mydirlabel ;
  char str[100],str2[100],str3[100];
  vector<TH1F*> h;
  if (dbe_) {
    dbe_->cd();
    dbe_->setCurrentFolder("HLT/Muon");
    snprintf(str2, 99, "%s",theL1CollectionLabel.encode().c_str() );
    mydirlabel = strtok(str2,"L1");
    snprintf(str3,99, "HLT/Muon/RateEfficiencies/%s",mydirlabel);
    dbe_->setCurrentFolder(str3);
    NumberOfEvents=dbe_->bookInt("NumberOfEvents");
    NumberOfL1Events=dbe_->bookInt("NumberOfL1Events");
    snprintf(str, 99, "%s",theL1CollectionLabel.encode().c_str() );
    mylabel = strtok(str,":");
    snprintf(chname, 255, "L1DR_%s", mylabel);
    snprintf(chtitle, 255, "L1 Gen association #DR, label=%s", mylabel);
    hL1DR= BookIt(chname, chtitle,theNbins, 0., 0.5);
    snprintf(chname, 255, "L2DR_%s", mylabel);
    snprintf(chtitle, 255, "L2 Gen association #DR, label=%s", mylabel);
    hL2DR= BookIt(chname, chtitle,theNbins, 0., 0.5);
    snprintf(chname, 255, "L3DR_%s", mylabel);
    snprintf(chtitle, 255, "L3 Gen association #DR, label=%s", mylabel);
    hL3DR= BookIt(chname, chtitle,theNbins, 0., 0.5);
    snprintf(chname, 255, "HLTSteps_%s", mylabel);
    snprintf(chtitle, 255, "Events passing the HLT filters, label=%s", mylabel);
    hSteps= BookIt(chname, chtitle,5, 0.5, 5.5);
    snprintf(chname, 255, "eff_%s", mylabel);
    snprintf(chtitle, 255, "Efficiency (%%) vs L1 Pt threshold (GeV), label=%s", mylabel);
    hL1eff = BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);

    snprintf(chname, 255, "rate_%s", mylabel);
    snprintf(chtitle, 255, "Rate (Hz) vs L1 Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);

    hL1rate =   BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);
    dbe_->cd();
    snprintf(str3,99, "HLT/Muon/Distributions/%s",mydirlabel);
    dbe_->setCurrentFolder(str3);
    snprintf(chname, 255, "pt_%s", mylabel);
    snprintf(chtitle, 255, "L1 Pt distribution label=%s", mylabel);
    hL1pt = BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);

    if (useMuonFromGenerator){
      snprintf(chtitle, 255, "L1 max ref pt efficiency label=%s", mylabel);
      snprintf(chname, 255, "MCTurnOn_%s", mylabel);
      hL1MCeff = BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);
      snprintf(chtitle, 255, "L1 max ref pt distribution label=%s", mylabel);
      snprintf(chname, 255, "MCptNorm_%s", mylabel);
      hMCptnor = BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);
      snprintf(chname, 255, "MCetaNorm_%s",mylabel);
      snprintf(chtitle, 255, "Norm  MC Eta ");
      hMCetanor = BookIt(chname, chtitle, 50, -2.1, 2.1);
      snprintf(chname, 255, "MCphiNorm_%s",mylabel);
      snprintf(chtitle, 255, "Norm  MC #Phi");
      hMCphinor = BookIt(chname, chtitle, 50, -3.15, 3.15);
      snprintf(chname, 255, "MCeta_%s", mylabel);
      snprintf(chtitle, 255, "L1 Eta distribution label=%s", mylabel);
      hL1etaMC = BookIt(chname, chtitle, 50, -2.1, 2.1);
      snprintf(chname, 255, "MCphi_%s", mylabel);
      snprintf(chtitle, 255, "L1 Phi distribution label=%s", mylabel);
      hL1phiMC = BookIt(chname, chtitle, 50, -3.15, 3.15);
    }
    if (useMuonFromReco){
      snprintf(chtitle, 255, "L1 max ref pt efficiency label=%s", mylabel);
      snprintf(chname, 255, "RECOTurnOn_%s", mylabel);
      hL1RECOeff = BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);
      snprintf(chtitle, 255, "L1 max reco ref pt distribution label=%s", mylabel);
      snprintf(chname, 255, "RECOptNorm_%s", mylabel);  
      hRECOptnor = BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);
      snprintf(chname, 255, "RECOetaNorm_%s",mylabel);
      snprintf(chtitle, 255, "Norm  RECO Eta ");
      hRECOetanor = BookIt(chname, chtitle, 50, -2.1, 2.1);
      snprintf(chname, 255, "RECOphiNorm_%s",mylabel);
      snprintf(chtitle, 255, "Norm  RECO #Phi");
      hRECOphinor = BookIt(chname, chtitle, 50, -3.15, 3.15);
      snprintf(chname, 255, "RECOeta_%s", mylabel);
      snprintf(chtitle, 255, "L1 Eta distribution label=%s", mylabel);
      hL1etaRECO = BookIt(chname, chtitle, 50, -2.1, 2.1);
      snprintf(chname, 255, "RECOphi_%s", mylabel);
      snprintf(chtitle, 255, "L1 Phi distribution label=%s", mylabel);
      hL1phiRECO = BookIt(chname, chtitle, 50, -3.15, 3.15);
    }
    for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
      dbe_->cd();
      snprintf(str3,99, "HLT/Muon/RateEfficiencies/%s",mydirlabel);
      dbe_->setCurrentFolder(str3);      
      snprintf(str, 99, "%s",theHLTCollectionLabels[i].encode().c_str() );
      mylabel = strtok(str,":");
      snprintf(chname, 255, "eff_%s",mylabel );
      snprintf(chtitle, 255, "Efficiency (%%) vs HLT Pt threshold (GeV), label=%s", mylabel);
      hHLTeff.push_back(BookIt(chname, chtitle, theNbins, thePtMin, thePtMax));
      snprintf(chname, 255, "rate_%s", mylabel);
      snprintf(chtitle, 255, "Rate (Hz) vs HLT Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
      hHLTrate.push_back(BookIt(chname, chtitle, theNbins, thePtMin, thePtMax));
      dbe_->cd();
      snprintf(str3,99, "HLT/Muon/Distributions/%s",mydirlabel);
      dbe_->setCurrentFolder(str3);
      snprintf(chname, 255, "pt_%s",mylabel );
      snprintf(chtitle, 255, "Pt distribution label=%s", mylabel); 
      hHLTpt.push_back(BookIt(chname, chtitle, theNbins, thePtMin, thePtMax));

      if (useMuonFromGenerator){
	snprintf(chtitle, 255, "Turn On curve, label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
	snprintf(chname, 255, "MCTurnOn_%s", mylabel);
	hHLTMCeff.push_back(BookIt(chname, chtitle, theNbins, thePtMin, thePtMax));   
	snprintf(chname, 255, "MCeta_%s",mylabel );
	snprintf(chtitle, 255, "Gen Eta Efficiency label=%s", mylabel);
	hHLTetaMC.push_back(BookIt(chname, chtitle, 50, -2.1, 2.1));
	snprintf(chname, 255, "MCphi_%s",mylabel );
	snprintf(chtitle, 255, "Gen Phi Efficiency label=%s", mylabel);
	hHLTphiMC.push_back(BookIt(chname, chtitle, 50, -3.15, 3.15));
      }
      if (useMuonFromReco){
	snprintf(chtitle, 255, "Turn On curve, label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
	snprintf(chname, 255, "RECOTurnOn_%s", mylabel);
	hHLTRECOeff.push_back(BookIt(chname, chtitle, theNbins, thePtMin, thePtMax));   
	snprintf(chname, 255, "RECOeta_%s",mylabel );
	snprintf(chtitle, 255, "Reco Eta Efficiency label=%s", mylabel);
	hHLTetaRECO.push_back(BookIt(chname, chtitle, 50, -2.1, 2.1));
	snprintf(chname, 255, "RECOphi_%s",mylabel );
	snprintf(chtitle, 255, "Reco Phi Efficiency label=%s", mylabel);
	hHLTphiRECO.push_back(BookIt(chname, chtitle, 50, -3.15, 3.15));
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

MonitorElement* HLTMuonGenericRate::BookIt(char* chname, char* chtitle, int Nbins, float Min, float Max) {
  LogWarning("HLTMuonVal")<<"Directory "<<dbe_->pwd()<<" Name "<<chname<<" Title:"<<chtitle;
  TH1F *h=new TH1F(chname, chtitle, Nbins, Min, Max);
  h->Sumw2();
  return dbe_->book1D(chname, h);
  delete h;
}

void HLTMuonGenericRate::WriteHistograms() {
  if (theRootFileName.size() != 0 && dbe_) dbe_->save(theRootFileName);
   return;
}

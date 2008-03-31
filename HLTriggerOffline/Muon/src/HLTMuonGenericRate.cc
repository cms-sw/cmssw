 /** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author M. Vander Donckt  (copied from J. Alcaraz)
 */

#include "HLTriggerOffline/Muon/interface/HLTMuonGenericRate.h"

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
  theNumberOfEvents = 0.;
  theNumberOfL1Events = 0.;
}

/// Destructor
HLTMuonGenericRate::~HLTMuonGenericRate(){
}

void HLTMuonGenericRate::analyze(const Event & event ){
  this_event_weight=1;
  ++theNumberOfEvents;
  LogDebug("HLTMuonVal")<<"In analyze for L1 trigger "<<theL1CollectionLabel<<" Event:"<<theNumberOfEvents;  
  // Get the muon with maximum pt at generator level or reconstruction, depending on the choice
  bool refmuon_found = false;
  bool refrecomuon_found = false;
  double ptuse = -1;
  double recoptuse = -1;
  if (useMuonFromGenerator) {
    Handle<HepMCProduct> genProduct;
    event.getByLabel(theGenLabel,genProduct);
    if (genProduct.failedToGet())return;
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
  Handle<reco::TrackCollection> muTracks;
  if (useMuonFromReco) {
    try {
      // Get the muon track collection from the event
      reco::TrackCollection::const_iterator muon;
      event.getByLabel(theRecoLabel.label(), muTracks);    
      if  ( muTracks.failedToGet() )return;
      for ( muon = muTracks->begin(); muon != muTracks->end(); ++muon ) {
	float pt1 = muon->pt();
	hRECOetanor->Fill(muon->eta());
	hRECOphinor->Fill(muon->phi());
	if (pt1>recoptuse) {
	  refrecomuon_found = true;
	  recoptuse = pt1;
	}
      }
    } catch (...) {
      // Do nothing
      LogWarning("HLTMuonVal")<<"NO Reco Collection";
     return; 
    }
  }

  
  if (ptuse > 0 ) hMCptnor->Fill(ptuse,this_event_weight);
  if (recoptuse > 0 ) hRECOptnor->Fill(recoptuse,this_event_weight);

  // Get the Trigger collection
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByLabel("triggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogWarning("HLTMuonVal") << "RAW-type HLT results not found, skipping event";
    return;
  }
  // Get the L1 collection
  /* Handle<TriggerFilterObjectWithRefs> l1mucands;
  event.getByLabel(theL1CollectionLabel, l1mucands);
  if (l1mucands.failedToGet()){
    LogDebug("HLTMuonVal")<<"No L1 Collection with label "<<theL1CollectionLabel;
    return; 
    } */
  vector<L1MuonParticleRef> l1cands;
  if ( triggerObj->filterIndex(theL1CollectionLabel.label())>=triggerObj->size() )return;
  triggerObj->getObjects(triggerObj->filterIndex(theL1CollectionLabel.label()),TriggerL1Mu,l1cands);
  ++theNumberOfL1Events;
 // Get the HLT collections
  unsigned hltsize=theHLTCollectionLabels.size();
  vector<vector<RecoChargedCandidateRef> > hltcands(hltsize);
  unsigned int modules_in_this_event = 0;
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
    if ( triggerObj->filterIndex(theHLTCollectionLabels[i].label())>=triggerObj->size() ) {
      LogDebug("HLTMuonVal")<<"No HLT Collection with label "<<theHLTCollectionLabels[i];
      break ;
    }
    triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[i].label()),TriggerMuon, hltcands[i]);
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
	  LogDebug("HLTMuonVal")<<"Filling L1 histos....";
	  if ( angularInfo.first < 999.){
	    hL1etaMC->Fill(angularInfo.first);
	    hL1phiMC->Fill(angularInfo.second);
	    LogDebug("HLTMuonVal")<<"Filling done";
	  }
	}
	if (useMuonFromReco){
	  pair<double,double> angularInfo=getRecoAngle(candref->eta(),candref->phi(), *muTracks );
	  LogDebug("HLTMuonVal")<<"Filling L1 histos....";
	  if ( angularInfo.first < 999.){
	    hL1etaRECO->Fill(angularInfo.first);
	    hL1phiRECO->Fill(angularInfo.second);
	    LogDebug("HLTMuonVal")<<"Filling done";
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
    unsigned int last_module = modules_in_this_event - 1;
    for (unsigned int i=0; i<=last_module; i++) {
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
			LogDebug("HLTMuonVal")<<"Filling HLT histos for MC ["<<i<<"]........";
			hHLTetaMC[i]->Fill(angularInfo.first);
			hHLTphiMC[i]->Fill(angularInfo.second);
			LogDebug("HLTMuonVal")<<"Filling done";
		      }
		    }
		    if (useMuonFromReco){
		      pair<double,double> angularInfo=getRecoAngle(candref->eta(),candref->phi(), *muTracks );
		      if ( angularInfo.first < 999.){
			LogDebug("HLTMuonVal")<<"Filling HLT histos for RECO....["<<i<<"]........";
			hHLTetaRECO[i]->Fill(angularInfo.first);
			hHLTphiRECO[i]->Fill(angularInfo.second);
			LogDebug("HLTMuonVal")<<"Filling done";
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

pair<double,double> HLTMuonGenericRate::getGenAngle(double eta, double phi, HepMC::GenEvent evt )
{

  LogDebug("HLTMuonVal")<< "in getGenAngle";
  double candDeltaR = 0.3;
  HepMC::GenEvent::particle_const_iterator part;
  HepMC::GenEvent::particle_const_iterator theAssociatedpart=evt.particles_end();
  pair<double,double> angle(999.,999.);
  LogDebug("HLTMuonVal")<< " candidate eta="<<eta<<" and phi="<<phi;
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
  LogDebug("HLTMuonVal")<< "Best deltaR="<<candDeltaR;
  return angle;

}
pair<double,double> HLTMuonGenericRate::getRecoAngle(double eta, double phi, reco::TrackCollection muTracks )
{

  LogDebug("HLTMuonVal")<< "in getRecoAngle";
  double candDeltaR = 0.3;
  reco::TrackCollection::const_iterator muon;
  reco::TrackCollection::const_iterator theAssociatedpart=muTracks.end();
  pair<double,double> angle(999.,999.);
  LogDebug("HLTMuonVal")<< " candidate eta="<<eta<<" and phi="<<phi;
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
  LogDebug("HLTMuonVal")<< "Best deltaR="<<candDeltaR;
  return angle;

}
void HLTMuonGenericRate::WriteHistograms(){
  // Write the histos to file
  ratedir->cd();
  hSteps->GetXaxis()->SetTitle("Trigger Filtering Step");
  hSteps->GetYaxis()->SetTitle("Events passing Trigger Step (%)");
  hSteps->Write();
  hL1eff->GetXaxis()->SetTitle("90% Muon Pt threshold (GeV)");
  hL1eff->Write();
  hL1rate->Write();
  hL1rate->SetMarkerStyle(20);
  hL1rate->GetXaxis()->SetTitle("90% Muon Pt threshold (GeV)");
  hL1rate->GetYaxis()->SetTitle("Rate (Hz)");
  if (useMuonFromGenerator){ 
    hL1MCeff->GetXaxis()->SetTitle("Generated Muon PtMax (GeV)");
    hL1MCeff->GetYaxis()->SetTitle("L1 trigger Efficiency (%)");
    hL1MCeff->Write();
  }
  if (useMuonFromReco){
    hL1RECOeff->GetXaxis()->SetTitle("Reconstructed Muon PtMax (GeV)");
    hL1RECOeff->GetYaxis()->SetTitle("L1 trigger Efficiency (%)");
    hL1RECOeff->Write();
  }
  distribdir->cd();
  hL1pt->GetXaxis()->SetTitle("Muon Pt (GeV)");
  hL1pt->Write();
  if (useMuonFromGenerator){
    hL1etaMC->GetXaxis()->SetTitle("Muon #eta");
    hL1etaMC->Write();
    hL1phiMC->GetXaxis()->SetTitle("Muon #phi");
    hL1phiMC->Write();
    hMCetanor->GetXaxis()->SetTitle("Gen Muon #eta");
    hMCetanor->Write();
    hMCphinor->GetXaxis()->SetTitle("Gen Muon #phi ");
    hMCphinor->Write();
  }
  if (useMuonFromReco){
    hL1etaRECO->GetXaxis()->SetTitle("Muon #eta");
    hL1etaRECO->Write();
    hL1phiRECO->GetXaxis()->SetTitle("Muon #phi");
    hL1phiRECO->Write();
    hRECOetanor->GetXaxis()->SetTitle("Reco Muon #eta");
    hRECOetanor->Write();
    hRECOphinor->GetXaxis()->SetTitle("Reco Muon #phi ");
    hRECOphinor->Write();
  }
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
    ratedir->cd();
    hHLTeff[i]->GetXaxis()->SetTitle("90% Muon Pt threshold (GeV)");
    hHLTeff[i]->Write();
    hHLTeff[i]->SetMarkerStyle(20);
    hHLTeff[i]->SetMarkerColor(i+2);
    hHLTeff[i]->SetLineColor(i+2);
    hHLTrate[i]->GetYaxis()->SetTitle("Rate (Hz)");
    hHLTrate[i]->SetMarkerStyle(20);
    hHLTrate[i]->SetMarkerColor(i+2);
    hHLTrate[i]->SetLineColor(i+2);
    hHLTrate[i]->GetXaxis()->SetTitle("90% Muon Pt threshold (GeV)");
    hHLTrate[i]->Write();
    distribdir->cd();
    hHLTpt[i]->GetXaxis()->SetTitle("HLT Muon Pt (GeV)");
    hHLTpt[i]->Write();
    if (useMuonFromGenerator){
      hHLTMCeff[i]->GetXaxis()->SetTitle("Generated Muon PtMax (GeV)");
      hHLTMCeff[i]->GetYaxis()->SetTitle("Trigger Efficiency (%)");
      hHLTMCeff[i]->Write();
      hHLTetaMC[i]->GetXaxis()->SetTitle("Gen Muon #eta");
      hHLTetaMC[i]->Write();
      hHLTphiMC[i]->GetXaxis()->SetTitle("Gen Muon #phi");
      hHLTphiMC[i]->Write();
    }
    if (useMuonFromReco){
      hHLTRECOeff[i]->GetXaxis()->SetTitle("Reconstructed Muon PtMax (GeV)");
      hHLTRECOeff[i]->GetYaxis()->SetTitle("Trigger Efficiency (%)");
      hHLTRECOeff[i]->Write();
      hHLTetaRECO[i]->GetXaxis()->SetTitle("Reco Muon #eta");
      hHLTetaRECO[i]->Write();
      hHLTphiRECO[i]->GetXaxis()->SetTitle("Reco Muon #phi");
      hHLTphiRECO[i]->Write();
    }
  }
  top->cd();
}
void HLTMuonGenericRate::BookHistograms(){
  char chname[256];
  char chtitle[256];
  char * mylabel ;
  char * mydirlabel ;
  char str[100],str2[100];
  TH1F *h;
  top=gDirectory;
  top->cd("RateEfficiencies");
  ratedir=gDirectory;
  snprintf(str2, 99, "%s",theL1CollectionLabel.encode().c_str() );
  mydirlabel = strtok(str2,"L1");
  ratedir->mkdir(mydirlabel);
  ratedir->cd(mydirlabel);
  ratedir=gDirectory;
  
  snprintf(str, 99, "%s",theL1CollectionLabel.encode().c_str() );
  mylabel = strtok(str,":");
  snprintf(chname, 255, "HLTSteps_%s", mylabel);
  snprintf(chtitle, 255, "Events passing the HLT filters, label=%s", mylabel);
  hSteps= new TH1F(chname, chtitle, 5, 0.5, 5.5);
  snprintf(chname, 255, "eff_%s", mylabel);
  snprintf(chtitle, 255, "Efficiency (%%) vs L1 Pt threshold (GeV), label=%s", mylabel);
  hL1eff = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  hL1eff->Sumw2();
  snprintf(chname, 255, "rate_%s", mylabel);
  snprintf(chtitle, 255, "Rate (Hz) vs L1 Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
  hL1rate = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  hL1rate->Sumw2();
  top->cd("Distributions");
  distribdir=gDirectory;
  distribdir->mkdir(mydirlabel);
  distribdir->cd(mydirlabel);
  distribdir=gDirectory;
  snprintf(chname, 255, "pt_%s", mylabel);
  snprintf(chtitle, 255, "L1 Pt distribution label=%s", mylabel);
  hL1pt = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  hL1pt->Sumw2();
  if (useMuonFromGenerator){
    snprintf(chtitle, 255, "L1 max ref pt efficiency label=%s", mylabel);
    snprintf(chname, 255, "MCTurnOn_%s", mylabel);
    hL1MCeff = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    hL1MCeff->Sumw2();
    snprintf(chtitle, 255, "L1 max ref pt distribution label=%s", mylabel);
    snprintf(chname, 255, "MCptNorm_%s", mylabel);
    hMCptnor = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    hMCptnor->Sumw2();
    snprintf(chname, 255, "MCetaNorm_%s",mylabel);
    snprintf(chtitle, 255, "Norm  MC Eta ");
    hMCetanor = new TH1F(chname, chtitle, 50, -2.1, 2.1);
    hMCetanor->Sumw2();
    snprintf(chname, 255, "MCphiNorm_%s",mylabel);
    snprintf(chtitle, 255, "Norm  MC #Phi");
    hMCphinor = new TH1F(chname, chtitle, 50, -3.15, 3.15);
    hMCphinor->Sumw2();
    snprintf(chname, 255, "MCeta_%s", mylabel);
    snprintf(chtitle, 255, "L1 Eta distribution label=%s", mylabel);
    hL1etaMC = new TH1F(chname, chtitle, 50, -2.1, 2.1);
    hL1etaMC->Sumw2();
    snprintf(chname, 255, "MCphi_%s", mylabel);
    snprintf(chtitle, 255, "L1 Phi distribution label=%s", mylabel);
    hL1phiMC = new TH1F(chname, chtitle, 50, -3.15, 3.15);
    hL1phiMC->Sumw2();
  }
  if (useMuonFromReco){
    snprintf(chtitle, 255, "L1 max ref pt efficiency label=%s", mylabel);
    snprintf(chname, 255, "RECOTurnOn_%s", mylabel);
    hL1RECOeff = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    hL1RECOeff->Sumw2();
    snprintf(chtitle, 255, "L1 max reco ref pt distribution label=%s", mylabel);
    snprintf(chname, 255, "RECOptNorm_%s", mylabel);  
    hRECOptnor = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    hRECOptnor->Sumw2();
    snprintf(chname, 255, "RECOetaNorm_%s",mylabel);
    snprintf(chtitle, 255, "Norm  RECO Eta ");
    hRECOetanor = new TH1F(chname, chtitle, 50, -2.1, 2.1);
    hRECOetanor->Sumw2();
    snprintf(chname, 255, "RECOphiNorm_%s",mylabel);
    snprintf(chtitle, 255, "Norm  RECO #Phi");
    hRECOphinor = new TH1F(chname, chtitle, 50, -3.15, 3.15);
    hRECOphinor->Sumw2();
    snprintf(chname, 255, "RECOeta_%s", mylabel);
    snprintf(chtitle, 255, "L1 Eta distribution label=%s", mylabel);
    hL1etaRECO = new TH1F(chname, chtitle, 50, -2.1, 2.1);
    hL1etaRECO->Sumw2();
    snprintf(chname, 255, "RECOphi_%s", mylabel);
    snprintf(chtitle, 255, "L1 Phi distribution label=%s", mylabel);
    hL1phiRECO = new TH1F(chname, chtitle, 50, -3.15, 3.15);
    hL1phiRECO->Sumw2();
  }
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
    ratedir->cd();
    snprintf(str, 99, "%s",theHLTCollectionLabels[i].encode().c_str() );
    mylabel = strtok(str,":");
    snprintf(chname, 255, "eff_%s",mylabel );
    snprintf(chtitle, 255, "Efficiency (%%) vs HLT Pt threshold (GeV), label=%s", mylabel);
    h=new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    h->Sumw2();
    hHLTeff.push_back(h);
    snprintf(chname, 255, "rate_%s", mylabel);
    snprintf(chtitle, 255, "Rate (Hz) vs HLT Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
    h=new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    h->Sumw2();
    hHLTrate.push_back(h);
    distribdir->cd();
    snprintf(chname, 255, "pt_%s",mylabel );
    snprintf(chtitle, 255, "Pt distribution label=%s", mylabel); 
    h=new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    h->Sumw2();
    hHLTpt.push_back(h);
    if (useMuonFromGenerator){
      snprintf(chtitle, 255, "Turn On curve, label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
      snprintf(chname, 255, "MCTurnOn_%s", mylabel);
      h=new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
      hHLTMCeff.push_back(h);   
      h->Sumw2();      
      snprintf(chname, 255, "MCeta_%s",mylabel );
      snprintf(chtitle, 255, "Gen Eta Efficiency label=%s", mylabel);
      h=new TH1F(chname, chtitle, 50, -2.1, 2.1);
      h->Sumw2();
      hHLTetaMC.push_back(h);
      snprintf(chname, 255, "MCphi_%s",mylabel );
      snprintf(chtitle, 255, "Gen Phi Efficiency label=%s", mylabel);
      h=new TH1F(chname, chtitle, 50, -3.15, 3.15);
      h->Sumw2();
      hHLTphiMC.push_back(h);
    }
    if (useMuonFromReco){
      snprintf(chtitle, 255, "Turn On curve, label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
      snprintf(chname, 255, "RECOTurnOn_%s", mylabel);
      h=new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
      h->Sumw2();      
      hHLTRECOeff.push_back(h);   
      snprintf(chname, 255, "RECOeta_%s",mylabel );
      snprintf(chtitle, 255, "Reco Eta Efficiency label=%s", mylabel);
      h=new TH1F(chname, chtitle, 50, -2.1, 2.1);
      h->Sumw2();      
      hHLTetaRECO.push_back(h);
      snprintf(chname, 255, "RECOphi_%s",mylabel );
      snprintf(chtitle, 255, "Reco Phi Efficiency label=%s", mylabel);
      h=new TH1F(chname, chtitle, 50, -3.15, 3.15);
      h->Sumw2();
      hHLTphiRECO.push_back(h);
    }
  }
  top->cd();
}


void HLTMuonGenericRate::FillHistograms(){
  // L1 operations
  for (unsigned int k=0; k<=theNbins+1; k++) {
      double this_eff = hL1eff->GetBinContent(k)/theNumberOfEvents;
      double this_eff_error = hL1eff->GetBinError(k)/theNumberOfEvents*sqrt(1-this_eff);
      double this_rate = theLuminosity*theCrossSection*this_eff;
      double this_rate_error = theLuminosity*theCrossSection*this_eff_error;
      hL1eff->SetBinContent(k,this_eff);
      hL1eff->SetBinError(k,this_eff_error);
      hL1rate->SetBinContent(k,this_rate);
      hL1rate->SetBinError(k,this_rate_error);
  }
  hL1eff->Scale(100.);
  hSteps->Scale(1./theNumberOfEvents);
  
  // HLT operations
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
      for (unsigned int k=0; k<=theNbins+1; k++) {
            double this_eff = hHLTeff[i]->GetBinContent(k)/theNumberOfL1Events;
            double this_eff_error = hHLTeff[i]->GetBinError(k)/theNumberOfL1Events;
            double this_rate = theLuminosity*theCrossSection*this_eff;
            double this_rate_error = theLuminosity*theCrossSection*this_eff_error;
            hHLTeff[i]->SetBinContent(k,this_eff);
            hHLTeff[i]->SetBinError(k,this_eff_error);
            hHLTrate[i]->SetBinContent(k,this_rate);
            hHLTrate[i]->SetBinError(k,this_rate_error);
      }
      hHLTeff[i]->Scale(100.);
      if (useMuonFromGenerator){
        TH1F *num=(TH1F*) hHLTMCeff[i]->Clone();
	hHLTMCeff[i]->Divide(num,hL1MCeff,1.,1.,"B");
	hHLTMCeff[i]->Scale(100.);
        num=(TH1F*) hHLTetaMC[i]->Clone();
	hHLTetaMC[i]->Divide(num,hL1etaMC,1.,1.,"B");
	hHLTetaMC[i]->Scale(100.);
        num=(TH1F*) hHLTphiMC[i]->Clone();
	hHLTphiMC[i]->Divide(num,hL1phiMC,1.,1.,"B");
	hHLTphiMC[i]->Scale(100.);
      }
      if (useMuonFromReco){
	TH1F *num=(TH1F*) hHLTRECOeff[i]->Clone();
	hHLTRECOeff[i]->Divide(num,hL1RECOeff,1.,1.,"B");
	hHLTRECOeff[i]->Scale(100.);
	num=(TH1F*) hHLTetaRECO[i]->Clone();
	hHLTetaRECO[i]->Divide(num,hL1etaRECO,1.,1.,"B");
	hHLTetaRECO[i]->Scale(100.);
	num=(TH1F*) hHLTphiRECO[i]->Clone();
	hHLTphiRECO[i]->Divide(num,hL1phiRECO,1.,1.,"B");
	hHLTphiRECO[i]->Scale(100.);
      }

  }
  if (useMuonFromGenerator){
    TH1F *num=(TH1F*) hL1MCeff->Clone();
    hL1MCeff->Divide(num,hMCptnor,1.,1.,"B");
    hL1MCeff->Scale(100.);
    num=(TH1F*) hL1etaMC->Clone();
    hL1etaMC->Divide(num,hMCetanor,1.,1.,"B");
    hL1etaMC->Scale(100.);
    num=(TH1F*) hL1phiMC->Clone();
    hL1phiMC->Divide(num,hMCphinor,1.,1.,"B");
    hL1phiMC->Scale(100.);
    }
  if (useMuonFromReco){
    TH1F *num=(TH1F*) hL1RECOeff->Clone();
    hL1RECOeff->Divide(num,hRECOptnor,1.,1.,"B");
    hL1RECOeff->Scale(100.);
    num=(TH1F*) hL1etaRECO->Clone();
    hL1etaRECO->Divide(num,hRECOetanor,1.,1.,"B");
    hL1etaRECO->Scale(100.);
    num=(TH1F*) hL1phiRECO->Clone();
    hL1phiRECO->Divide(num,hRECOphinor,1.,1.,"B");
    hL1phiRECO->Scale(100.);
  }

}


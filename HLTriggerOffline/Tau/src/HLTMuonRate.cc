/** \class HLTMuonRate
 *  Get L1/HLT efficiency/rate plots
 *  \author Sho Maruyama  (copied from J. Alcaraz)
 */
#include "HLTriggerOffline/Tau/interface/HLTMuonRate.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
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
using namespace math;
using namespace reco;
using namespace trigger;
using namespace l1extra;
typedef std::vector< edm::ParameterSet > Parameters;
HLTMuonRate::HLTMuonRate(const ParameterSet& pset, int Index)
{
  InputLabel = pset.getUntrackedParameter<InputTag>("InputLabel");
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
HLTMuonRate::~HLTMuonRate(){}

void HLTMuonRate::analyze(const Event & event ){
  this_event_weight=1;
  NumberOfEvents->Fill(++theNumberOfEvents);
  LogTrace("HLTMuonVal")<<"In analyze for L1 trigger "<<theL1CollectionLabel<<" Event:"<<theNumberOfEvents;  
  bool refmuon_found = false;
  double ptuse = -1;
    Handle<vector<XYZTLorentzVectorD> > refVector;
    event.getByLabel(InputLabel,refVector);
for(unsigned int i = 0; i < refVector->size(); i++){
      double pt1 = refVector->at(i).pt();
      hEtaNor->Fill(refVector->at(i).eta());
      hPhiNor->Fill(refVector->at(i).phi());
      if (pt1>ptuse) {
        refmuon_found = true;
        ptuse = pt1;
      }
    } 
  if (ptuse > 0 ) hPtNor->Fill(ptuse,this_event_weight);
  // Get the L1 collection
  Handle<TriggerFilterObjectWithRefs> l1candsHandle;
  event.getByLabel(theL1CollectionLabel, l1candsHandle);
  // Get the HLT collections
  std::vector<Handle<TriggerFilterObjectWithRefs> > hltcandsHandle(theHLTCollectionLabels.size());
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
      event.getByLabel(theHLTCollectionLabels[i], hltcandsHandle[i]);
      if (hltcandsHandle[i].failedToGet()) break;
  }
  vector<L1MuonParticleRef> l1cands;
  l1candsHandle->getObjects(81,l1cands);
  unsigned hltsize=theHLTCollectionLabels.size();
  vector<vector<RecoChargedCandidateRef> > hltcands(hltsize);
  unsigned int modules_in_this_event = 0;
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
    hltcandsHandle[i]->getObjects(93, hltcands[i]);
    modules_in_this_event++;
  }
  // Fix L1 thresholds to obtain HLT plots
  unsigned int nL1FoundRef = 0;
  double epsilon = 0.001;
  for (unsigned int k=0; k<l1cands.size(); k++) {
    double ptLUT = l1cands.at(k)->pt();
    // Add "epsilon" to avoid rounding errors when ptLUT==L1Threshold
    if (ptLUT+epsilon>theL1ReferenceThreshold) {
      nL1FoundRef++;
      hL1pt->Fill(ptLUT);
    pair<double,double> angularInfo=getAngle(l1cands.at(k)->eta(),l1cands.at(k)->phi(), refVector );
    LogTrace("HLTMuonVal")<<"Filling L1 histos....";
    if ( angularInfo.first < 999.){
      hL1Eta->Fill(angularInfo.first);
      hL1Phi->Fill(angularInfo.second);
          float dphi=angularInfo.second-l1cands.at(k)->phi();
          float deta=angularInfo.first-l1cands.at(k)->eta();
          hL1DR->Fill(sqrt(dphi*dphi+deta*deta));
      LogTrace("HLTMuonVal")<<"Filling done";
    }
    }
  }
  if (nL1FoundRef>=theNumberOfObjects){
    hSteps->Fill(1.);  
  }
  if (ptuse>0){
    int last_module = modules_in_this_event - 1;
    for ( int i=0; i<=last_module; i++) {
      double ptcut = theHLTReferenceThreshold;
      unsigned nFound = 0;
      for (unsigned int k=0; k<hltcands[i].size(); k++) {
    RecoChargedCandidateRef candref = RecoChargedCandidateRef(hltcands[i][k]);
    reco::TrackRef tk = candref->get<reco::TrackRef>();
    double pt = tk->pt();
    if (pt>ptcut) nFound++;
      }
      if (nFound>=theNumberOfObjects){
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
    reco::TrackRef tk = candref->get<reco::TrackRef>();
    double pt = tk->pt();
    if ( ptcut == thePtMin ) {
      hHLTpt[i]->Fill(pt);
        pair<double,double> angularInfo=getAngle(candref->eta(),candref->phi(), refVector );
        if ( angularInfo.first < 999.){
          float dphi=angularInfo.second-candref->phi();
          float deta=angularInfo.first-candref->eta();
          float d=sqrt(dphi*dphi+deta*deta);
          if (i==0)hL2DR->Fill(d);
          if ((modules_in_this_event==2 && i==1)||i==2 )hL3DR->Fill(d);
          LogTrace("HLTMuonVal")<<"Filling done";
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

pair<double,double> HLTMuonRate::getAngle(double eta, double phi, Handle< vector<XYZTLorentzVectorD> > & refVector)
{
  LogTrace("HLTMuonVal")<< "in getAngle";
  double candDeltaR = 0.4;
  pair<double,double> angle(999.,999.);
  LogTrace("HLTMuonVal")<< " candidate eta="<<eta<<" and phi="<<phi;
  for (unsigned int i = 0; i < refVector->size(); i++ ) {
      double Deta=eta - refVector->at(i).eta();
      double Dphi=phi - refVector->at(i).phi();
      double deltaR = sqrt(Deta*Deta+Dphi*Dphi);
      if ( deltaR < candDeltaR ) {
    candDeltaR=deltaR;
        angle.first  = refVector->at(i).eta();
        angle.second = refVector->at(i).phi();
    }
  }
  LogTrace("HLTMuonVal")<< "Best deltaR="<<candDeltaR;
  return angle;
}

void HLTMuonRate::BookHistograms(){
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
    snprintf(chtitle, 255, "L1 association #DR, label=%s", mylabel);
    hL1DR= BookIt(chname, chtitle,theNbins, 0., 0.5);
    snprintf(chname, 255, "L2DR_%s", mylabel);
    snprintf(chtitle, 255, "L2 association #DR, label=%s", mylabel);
    hL2DR= BookIt(chname, chtitle,theNbins, 0., 0.5);
    snprintf(chname, 255, "L3DR_%s", mylabel);
    snprintf(chtitle, 255, "L3 association #DR, label=%s", mylabel);
    hL3DR= BookIt(chname, chtitle,theNbins, 0., 0.5);
    snprintf(chname, 255, "HLTSteps_%s", mylabel);
    snprintf(chtitle, 255, "Events passing the HLT filters, label=%s", mylabel);
    hSteps= BookIt(chname, chtitle,5, 0.5, 5.5);
    snprintf(chname, 255, "eff_%s", mylabel);
    snprintf(chtitle, 255, "Efficiency (%%) vs L1 Pt threshold (GeV/c), label=%s", mylabel);
    hL1eff = BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);
    snprintf(chname, 255, "rate_%s", mylabel);
    snprintf(chtitle, 255, "Rate (Hz) vs L1 Pt threshold (GeV/c), label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
    hL1rate =   BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);
    dbe_->cd();
    snprintf(str3,99, "HLT/Muon/Distributions/%s",mydirlabel);
    dbe_->setCurrentFolder(str3);
    snprintf(chname, 255, "pt_%s", mylabel);
    snprintf(chtitle, 255, "L1 Pt distribution label=%s", mylabel);
    hL1pt = BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);
      snprintf(chtitle, 255, "L1 max ref pt distribution label=%s", mylabel);
      snprintf(chname, 255, "ptNorm_%s", mylabel);
      hPtNor = BookIt(chname, chtitle, theNbins, thePtMin, thePtMax);
      snprintf(chname, 255, "etaNorm_%s",mylabel);
      snprintf(chtitle, 255, "Norm  #eta ");
      hEtaNor = BookIt(chname, chtitle, 50, -2.5, 2.5);
      snprintf(chname, 255, "phiNorm_%s",mylabel);
      snprintf(chtitle, 255, "Norm  #phi");
      hPhiNor = BookIt(chname, chtitle, 50, -3.3, 3.3);
      snprintf(chname, 255, "eta_%s", mylabel);
      snprintf(chtitle, 255, "L1 Eta distribution label=%s", mylabel);
      hL1Eta = BookIt(chname, chtitle, 50, -2.5, 2.5);
      snprintf(chname, 255, "phi_%s", mylabel);
      snprintf(chtitle, 255, "L1 Phi distribution label=%s", mylabel);
      hL1Phi = BookIt(chname, chtitle, 50, -3.3, 3.3);
    for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
      dbe_->cd();
      snprintf(str3,99, "HLT/Muon/RateEfficiencies/%s",mydirlabel);
      dbe_->setCurrentFolder(str3);      
      snprintf(str, 99, "%s",theHLTCollectionLabels[i].encode().c_str() );
      mylabel = strtok(str,":");
      snprintf(chname, 255, "eff_%s",mylabel );
      snprintf(chtitle, 255, "Efficiency (%%) vs HLT Pt threshold (GeV/c), label=%s", mylabel);
      hHLTeff.push_back(BookIt(chname, chtitle, theNbins, thePtMin, thePtMax));
      snprintf(chname, 255, "rate_%s", mylabel);
      snprintf(chtitle, 255, "Rate (Hz) vs HLT Pt threshold (GeV/c), label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
      hHLTrate.push_back(BookIt(chname, chtitle, theNbins, thePtMin, thePtMax));
      dbe_->cd();
      snprintf(str3,99, "HLT/Muon/Distributions/%s",mydirlabel);
      dbe_->setCurrentFolder(str3);
      snprintf(chname, 255, "pt_%s",mylabel );
      snprintf(chtitle, 255, "Pt distribution label=%s", mylabel); 
      hHLTpt.push_back(BookIt(chname, chtitle, theNbins, thePtMin, thePtMax));
    }
    hSteps->setAxisTitle("Trigger Filtering Step");
    hSteps->setAxisTitle("Events passing Trigger Step",2);
    hL1eff->setAxisTitle("90% Muon Pt threshold (GeV/c)");
    hL1rate->setAxisTitle("90% Muon Pt threshold (GeV/c)");
    hL1rate->setAxisTitle("Rate (Hz)",2);
    hL1pt->setAxisTitle("Muon Pt (GeV/c)");
    for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
      hHLTeff[i]->setAxisTitle("90% Muon Pt threshold (GeV/c)");
      hHLTrate[i]->setAxisTitle("Rate (Hz)",2);
      hHLTrate[i]->setAxisTitle("90% Muon Pt threshold (GeV/c)",1);
      hHLTpt[i]->setAxisTitle("HLT Muon Pt (GeV/c)",1);
    }
  }
}

MonitorElement* HLTMuonRate::BookIt(char* chname, char* chtitle, int Nbins, float Min, float Max) {
  LogDebug("HLTMuonVal")<<"Directory "<<dbe_->pwd()<<" Name "<<chname<<" Title:"<<chtitle;
  TH1F *h = new TH1F(chname, chtitle, Nbins, Min, Max);
  h->Sumw2();
  return dbe_->book1D(chname, h);
  delete h;
}

void HLTMuonRate::WriteHistograms() {
  if (theRootFileName.size() != 0 && dbe_) dbe_->save(theRootFileName);
   return;
}

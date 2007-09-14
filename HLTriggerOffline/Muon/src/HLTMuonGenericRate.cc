 /** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author M. Vander Donckt  (copied from J. Alcaraz)
 */

#include "HLTriggerOffline/Muon/interface/HLTMuonGenericRate.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
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
typedef std::vector< edm::ParameterSet > Parameters;

/// Constructor
HLTMuonGenericRate::HLTMuonGenericRate(const ParameterSet& pset, int Index)
{
  useMuonFromGenerator = pset.getParameter<bool>("UseMuonFromGenerator");
  if(useMuonFromGenerator)theGenLabel = pset.getUntrackedParameter<InputTag>("GenLabel");
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
}

/// Destructor
HLTMuonGenericRate::~HLTMuonGenericRate(){
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
  snprintf(chname, 255, "eff_%s", mylabel);
  snprintf(chtitle, 255, "Efficiency (%%) vs L1 Pt threshold (GeV), label=%s", mylabel);
  hL1eff = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  snprintf(chname, 255, "rate_%s", mylabel);
  snprintf(chtitle, 255, "Rate (Hz) vs L1 Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
  hL1rate = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  top->cd("Distributions");
  distribdir=gDirectory;
  distribdir->mkdir(mydirlabel);
  distribdir->cd(mydirlabel);
  distribdir=gDirectory;
  snprintf(chtitle, 255, "L1 max ref pt efficiency label=%s", mylabel);
  snprintf(chname, 255, "MCeff_%s", mylabel);
  hL1MCeff = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  snprintf(chtitle, 255, "L1 max ref pt distribution label=%s", mylabel);
  snprintf(chname, 255, "MCptNorm_%s", mylabel);
  hMCptnor = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  snprintf(chname, 255, "MCetaNorm_%s",mylabel);
  snprintf(chtitle, 255, "Norm  MC Eta ");
  hMCetanor = new TH1F(chname, chtitle, 50, -2.5, 2.5);
  snprintf(chname, 255, "MCphiNorm_%s",mylabel);
  snprintf(chtitle, 255, "Norm  MC Phi");
  hMCphinor = new TH1F(chname, chtitle, 50, -3.15, 3.15);
  snprintf(chname, 255, "pt_%s", mylabel);
  snprintf(chtitle, 255, "L1 Pt distribution label=%s", mylabel);
  hL1pt = new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
  snprintf(chname, 255, "eta_%s", mylabel);
  snprintf(chtitle, 255, "L1 Eta distribution label=%s", mylabel);
  hL1eta = new TH1F(chname, chtitle, 50, -2.5, 2.5);
  snprintf(chname, 255, "phi_%s", mylabel);
  snprintf(chtitle, 255, "L1 Phi distribution label=%s", mylabel);
  hL1phi = new TH1F(chname, chtitle, 50, -3.15, 3.15);
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
    ratedir->cd();
    snprintf(str, 99, "%s",theHLTCollectionLabels[i].encode().c_str() );
    mylabel = strtok(str,":");
    snprintf(chname, 255, "eff_%s",mylabel );
    snprintf(chtitle, 255, "Efficiency (%%) vs HLT Pt threshold (GeV), label=%s", mylabel);
    h=new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    hHLTeff.push_back(h);
    snprintf(chname, 255, "rate_%s", mylabel);
    snprintf(chtitle, 255, "Rate (Hz) vs HLT Pt threshold (GeV), label=%s, L=%.2E (cm^{-2} s^{-1})", mylabel, theLuminosity*1.e33);
    h=new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    hHLTrate.push_back(h);
    distribdir->cd();
    snprintf(chname, 255, "MCeff_%s", mylabel);
    hHLTMCeff.push_back(new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax));
    snprintf(chname, 255, "pt_%s",mylabel );
    snprintf(chtitle, 255, "Pt distribution label=%s", mylabel); 
    h=new TH1F(chname, chtitle, theNbins, thePtMin, thePtMax);
    hHLTpt.push_back(h);
    snprintf(chname, 255, "eta_%s",mylabel );
    snprintf(chtitle, 255, "Eta Efficiency label=%s", mylabel);
    h=new TH1F(chname, chtitle, 50, -2.5, 2.5);
    h->Sumw2();
    hHLTeta.push_back(h);
    snprintf(chname, 255, "phi_%s",mylabel );
    snprintf(chtitle, 255, "Phi Efficiency label=%s", mylabel);
    h=new TH1F(chname, chtitle, 50, -3.15, 3.15);
    h->Sumw2();
    hHLTphi.push_back(h);
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
  hL1MCeff->Divide(hMCptnor);
  hL1MCeff->Scale(100.);
  if (useMuonFromGenerator){
    hL1eta->Divide(hMCetanor);
    hL1eta->Scale(100.);
    hL1phi->Divide(hMCphinor);
    hL1phi->Scale(100.);
  }
  
  // HLT operations
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
      for (unsigned int k=0; k<=theNbins+1; k++) {
            double this_eff = hHLTeff[i]->GetBinContent(k)/theNumberOfEvents;
            double this_eff_error = hHLTeff[i]->GetBinError(k)/theNumberOfEvents;
            double this_rate = theLuminosity*theCrossSection*this_eff;
            double this_rate_error = theLuminosity*theCrossSection*this_eff_error;
            hHLTeff[i]->SetBinContent(k,this_eff);
            hHLTeff[i]->SetBinError(k,this_eff_error);
            hHLTrate[i]->SetBinContent(k,this_rate);
            hHLTrate[i]->SetBinError(k,this_rate_error);
      }
      hHLTeff[i]->Scale(100.);
      hHLTMCeff[i]->Divide(hMCptnor);
      hHLTMCeff[i]->Scale(100.);
      if (useMuonFromGenerator){
	hHLTeta[i]->Divide(hMCetanor);
	hHLTeta[i]->Scale(100.);
	hHLTphi[i]->Divide(hMCphinor);
	hHLTphi[i]->Scale(100.);
      }
  }

}
void HLTMuonGenericRate::WriteHistograms(){
  // Write the histos to file
  ratedir->cd();
  hL1eff->GetXaxis()->SetTitle("90% Muon Pt threshold (GeV)");
  hL1eff->Write();
  hL1rate->Write();
  hL1rate->SetMarkerStyle(20);
  hL1rate->GetXaxis()->SetTitle("90% Muon Pt threshold (GeV)");
  hL1rate->GetYaxis()->SetTitle("Rate (Hz)");
  distribdir->cd();
  hL1pt->GetXaxis()->SetTitle("Muon Pt (GeV)");
  hL1pt->Write();
  hL1eta->GetXaxis()->SetTitle("Muon eta");
  hL1eta->Write();
  hL1phi->GetXaxis()->SetTitle("Muon phi");
  hL1phi->Write();
  hL1MCeff->GetXaxis()->SetTitle("Generated Muon PtMax (GeV)");
  hL1MCeff->GetYaxis()->SetTitle("L1 trigger Efficiency (%)");
  hL1MCeff->Write();
  if (useMuonFromGenerator){
    hMCetanor->GetXaxis()->SetTitle("Gen Muon Eta");
    hMCetanor->Write();
    hMCphinor->GetXaxis()->SetTitle("Gen Muon Phi ");
    hMCphinor->Write();
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
    hHLTpt[i]->GetXaxis()->SetTitle("Muon Pt (GeV)");
    hHLTpt[i]->Write();
    hHLTeta[i]->GetXaxis()->SetTitle("Muon eta");
    hHLTeta[i]->Write();
    hHLTphi[i]->GetXaxis()->SetTitle("Muon phi");
    hHLTphi[i]->Write();
    hHLTMCeff[i]->GetXaxis()->SetTitle("Generated Muon PtMax (GeV)");
    hHLTMCeff[i]->GetYaxis()->SetTitle("Trigger Efficiency (%)");
    hHLTMCeff[i]->Write();
  }
  top->cd();
}

void HLTMuonGenericRate::analyze(const Event & event ){
  this_event_weight=1;
  ++theNumberOfEvents;
  LogDebug("HLTMuonVal")<<"In analyze for L1 trigger "<<theL1CollectionLabel<<" Event:"<<theNumberOfEvents;  
  // Get the muon with maximum pt at generator level or reconstruction, depending on the choice
  bool refmuon_found = false;
  double ptuse = -1;
  if (useMuonFromGenerator) {
    Handle<HepMCProduct> genProduct;
    event.getByLabel(theGenLabel,genProduct);
    evt= genProduct->GetEvent();
    if ( evt == NULL ) return;
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

  // Get the L1 collection
  Handle<HLTFilterObjectWithRefs> l1cands;
  try {
      event.getByLabel(theL1CollectionLabel, l1cands);
  } catch (...) {
      // Do nothing
      return;
  }

  // Get the HLT collections
  std::vector<Handle<HLTFilterObjectWithRefs> > hltcands;
  hltcands.reserve(theHLTCollectionLabels.size());
  unsigned int modules_in_this_event = 0;
  for (unsigned int i=0; i<theHLTCollectionLabels.size(); i++) {
      try {
            event.getByLabel(theHLTCollectionLabels[i], hltcands[i]);
      } catch (...) {
            break;
      }
      modules_in_this_event++;
  }
  if (!useMuonFromGenerator) {
      unsigned int i=modules_in_this_event-1;
      for (unsigned int k=0; k<hltcands[i]->size(); k++) {
	RefToBase<Candidate> candref = hltcands[i]->getParticleRef(k);
            TrackRef tk = candref->get<TrackRef>();
            double pt = tk->pt();
            if (pt>ptuse) {
	      refmuon_found = true;
	      ptuse = pt;
            }
      }
  }
  if (ptuse > 0 ) hMCptnor->Fill(ptuse,this_event_weight);

  if (useMuonFromGenerator) theAssociatedGenPart=evt->particles_end(); 
  // Fix L1 thresholds to obtain HLT plots
  unsigned int nL1FoundRef = 0;
  double epsilon = 0.001;
  for (unsigned int k=0; k<l1cands->size(); k++) {
    RefToBase<Candidate> candref = l1cands->getParticleRef(k);
    //LogDebug("HLTMuonVal") << " returned "<<(*theAssociatedGenPart)->momentum().eta()<<std::endl;
    // L1 PTs are "quantized" due to LUTs. 
    // Their meaning: true_pt > ptLUT more than 90% pof the times
    double ptLUT = candref->pt();
    // Add "epsilon" to avoid rounding errors when ptLUT==L1Threshold
    if (ptLUT+epsilon>theL1ReferenceThreshold) {
	nL1FoundRef++;
	hL1pt->Fill(ptLUT);
	if (useMuonFromGenerator){
	  pair<double,double> angularInfo=getGenAngle(candref, *evt );
	  LogDebug("HLTMuonVal")<<"Filling L1 histos....";
	  hL1eta->Fill(angularInfo.first);
	  hL1phi->Fill(angularInfo.second);
	  LogDebug("HLTMuonVal")<<"Filling done";
	} else {
	  hL1eta->Fill(candref->eta());
	  hL1phi->Fill(candref->phi());
	}
    }
  }
  if (nL1FoundRef>=theNumberOfObjects && ptuse>0) hL1MCeff->Fill(ptuse,this_event_weight);

  if (ptuse>0){
    unsigned int last_module = modules_in_this_event - 1;
    if ((!useMuonFromGenerator) && last_module>0) last_module--;
    for (unsigned int i=0; i<=last_module; i++) {
      double ptcut = theHLTReferenceThreshold;
      unsigned nFound = 0;
      for (unsigned int k=0; k<hltcands[i]->size(); k++) {
	RefToBase<Candidate> candref = hltcands[i]->getParticleRef(k);
	TrackRef tk = candref->get<TrackRef>();
	double pt = tk->pt();
	if (pt>ptcut) nFound++;
      }
      if (nFound>=theNumberOfObjects) hHLTMCeff[i]->Fill(ptuse,this_event_weight);
    }
  }

  for (unsigned int j=0; j<theNbins; j++) {
      double ptcut = thePtMin + j*(thePtMax-thePtMin)/theNbins;

      // L1 filling
      unsigned int nFound = 0;
      for (unsigned int k=0; k<l1cands->size(); k++) {
            RefToBase<Candidate> candref = l1cands->getParticleRef(k);
            double pt = candref->pt();
            if (pt>ptcut) nFound++;
      }
      if (nFound>=theNumberOfObjects) hL1eff->Fill(ptcut,this_event_weight);

      // Stop here if L1 reference cuts were not satisfied
      if (nL1FoundRef<theNumberOfObjects) continue;

      // HLT filling
      for (unsigned int i=0; i<modules_in_this_event; i++) {
            unsigned nFound = 0;
            for (unsigned int k=0; k<hltcands[i]->size(); k++) {
                  RefToBase<Candidate> candref = hltcands[i]->getParticleRef(k);
                  TrackRef tk = candref->get<TrackRef>();
                  double pt = tk->pt();
                  if ( ptcut == thePtMin ) {
		    hHLTpt[i]->Fill(pt);
		    if (useMuonFromGenerator){
		      pair<double,double> angularInfo=getGenAngle(candref, *evt );
		      LogDebug("HLTMuonVal")<<"Filling HLT histos ["<<i<<"]........";
		      hHLTeta[i]->Fill(angularInfo.first);
		      hHLTphi[i]->Fill(angularInfo.second);
		      LogDebug("HLTMuonVal")<<"Filling done";
		    }else{
		      hHLTeta[i]->Fill(candref->eta());
		      hHLTphi[i]->Fill(candref->pt());
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

pair<double,double> HLTMuonGenericRate::getGenAngle(RefToBase<Candidate> candref, HepMC::GenEvent evt )
{

  LogDebug("HLTMuonVal")<< "in getGenAngle";
  double candDeltaR = 999.0;
  HepMC::GenEvent::particle_const_iterator part;
  HepMC::GenEvent::particle_const_iterator theAssociatedpart=evt.particles_end();
  pair<double,double> angle(999.,999.);
  double eta=candref->eta();
  double phi=candref->phi();
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

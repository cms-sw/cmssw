

// -*- C++ -*-
//
// Package:    L25TauAnalyzer
// Class:      L25TauAnalyzer
// 
/**\class L25TauAnalyzer L25TauAnalyzer.cc HLTriggerOffline/Tau/src/L25TauAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Eduardo Luiggi
//         Created:  Fri Apr  4 16:37:44 CDT 2008
// $Id: L25TauAnalyzer.cc,v 1.8 2009/12/18 20:44:55 wmtan Exp $
//
//

#include "HLTriggerOffline/Tau/interface/L25TauAnalyzer.h"
#include "Math/GenVector/VectorUtil.h"

// system include files
using namespace edm;
using namespace reco;
using namespace std;

L25TauAnalyzer::L25TauAnalyzer(const edm::ParameterSet& iConfig){
  jetTagSrc_ = iConfig.getParameter<InputTag>("JetTagProd");
  jetMCTagSrc_ = iConfig.getParameter<InputTag>("JetMCTagProd");
  rootFile_ = iConfig.getParameter<string>("outputFileName");
  signal_ = iConfig.getParameter<bool>("Signal");
  minTrackPt_ =iConfig.getParameter<double>("MinTrackPt");
  signalCone_ =iConfig.getParameter<double>("SignalCone");
  isolationCone_ =iConfig.getParameter<double>("IsolationCone");
 
  l25file = new TFile(rootFile_.c_str(),"recreate");
  l25tree = new TTree("l25tree","Level 2.5 Tau Tree");

  jetEt=0.;
  jetEta=0.;
  jetMCEt=0.;
  jetMCEta=0.;
  leadSignalTrackPt=0.;
  trkDrRMS=0.;
  trkDrRMSA=0.;
  leadTrkJetDeltaR=0.;
  numPixTrkInJet=0;
  numQPixTrkInJet=0;
  numQPixTrkInSignalCone=0;
  numQPixTrkInAnnulus=0;
  hasLeadTrk=0;
  emf=0;


  l25tree->Branch("jetEt", &jetEt, "jetEt/F");
  l25tree->Branch("jetEMF", &emf, "jetEMF/F");
  l25tree->Branch("jetEta", &jetEta, "jetEta/F");
  l25tree->Branch("jetMCEt", &jetMCEt, "jetMCEt/F");
  l25tree->Branch("jetMCEta", &jetMCEta, "jetMCEta/F");
  l25tree->Branch("leadTrackPt", &leadSignalTrackPt, "leadTrackPt/F");
  l25tree->Branch("trackDeltaRRMS", &trkDrRMS, "trackDeltaRRMS/F");
  l25tree->Branch("trackDeltaRRMSAll", &trkDrRMSA, "trackDeltaRRMSAll/F");
  l25tree->Branch("matchingCone", &leadTrkJetDeltaR, "matchingCone/F");
  l25tree->Branch("nTracks", &numPixTrkInJet, "nTracks/I");
  l25tree->Branch("nQTracks", &numQPixTrkInJet, "nQTracks/I");
  l25tree->Branch("nQTracksInSignal", &numQPixTrkInSignalCone, "nQTracksInSignal/I");
  l25tree->Branch("nQTracksInAnnulus", &numQPixTrkInAnnulus, "nQTracksInAnnulus/I");
  l25tree->Branch("hasLeadTrack", &hasLeadTrk, "hasLeadTrack/B");
}


L25TauAnalyzer::~L25TauAnalyzer(){

}




void L25TauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){


   Handle<IsolatedTauTagInfoCollection> tauTagInfo;
   using namespace reco;
   Handle<LVColl> mcInfo;							 

   if(signal_){ 								 
      iEvent.getByLabel(jetMCTagSrc_, mcInfo);				 
   } 									 
   

   //get isolation tag
   if(iEvent.getByLabel(jetTagSrc_, tauTagInfo))
     {
       if(tauTagInfo->size()>0)
	 for(IsolatedTauTagInfoCollection::const_iterator i = tauTagInfo->begin();i!=tauTagInfo->end();++i)
	   {
	     MatchElementL25 m;
	     m.matched=false;
	     m.mcEt=0;
	     m.mcEta=0;
	     m.deltar=0;

	     if(signal_)
	       {
		 if(iEvent.getByLabel(jetMCTagSrc_,mcInfo))
		   m=match(*(i->jet()),*mcInfo);
	       }

	     if((signal_&&m.matched)||(!signal_))
	       {

		 const Jet* Jet =i->jet().get();
		 const CaloJet* calojet = dynamic_cast<const CaloJet*>(Jet);
		 emf = calojet->emEnergyFraction();

		 
		 jetEt=i->jet()->et();
		 jetEta=i->jet()->eta();
		 jetMCEt=m.mcEt;
		 jetMCEta=m.mcEta;
		 numPixTrkInJet = i->allTracks().size();
		 numQPixTrkInJet = i->selectedTracks().size();
		 trkDrRMS =trackDrRMS(*i,i->selectedTracks());
		 trkDrRMSA =trackDrRMS(*i,i->allTracks());
 

		 //Search Leading Track
		 const TrackRef leadTk = i->leadingSignalTrack();
		 if(!leadTk)
		   {
		     numQPixTrkInSignalCone=0;
		     numQPixTrkInAnnulus=0;
		     leadSignalTrackPt=0;
		     leadTrkJetDeltaR=0;
		     hasLeadTrk=false;
		   }
		 else
		   {
		     leadTrkJetDeltaR = ROOT::Math::VectorUtil::DeltaR(i->jet()->p4().Vect(), leadTk->momentum());		      	      
    		     leadSignalTrackPt=leadTk->pt();
		     numQPixTrkInSignalCone=(i->tracksInCone(leadTk->momentum(), signalCone_, minTrackPt_)).size();
		     numQPixTrkInAnnulus=(i->tracksInCone(leadTk->momentum(), isolationCone_, minTrackPt_)).size()-numQPixTrkInSignalCone;
		     hasLeadTrk=true;
		   }
	       }
	      l25tree->Fill();
	   }
     }
}




float 
L25TauAnalyzer::trackDrRMS(const reco::IsolatedTauTagInfo& info,const TrackRefVector& tracks)
{
  float rms = 0.;
  float sumet = 0.;
  
  
  //First find the weighted track
  if(tracks.size()>0)
    {
      math::XYZVector center = tracks[0]->momentum();

      for(size_t i = 1;i<tracks.size();++i)
	{
	  center+=tracks[i]->momentum();
	}

      //Now calculate DeltaR

      for(size_t i = 0;i<tracks.size();++i)
	{
	  rms+= tracks[i]->pt()*pow(ROOT::Math::VectorUtil::DeltaR(tracks[i]->momentum(),center),2);
	  sumet+=tracks[i]->pt();
	}
      
    }
     
  if(sumet==0.)
    sumet=1;
  
  return rms/sumet;
}



MatchElementL25 
L25TauAnalyzer::match(const reco::Jet& jet,const LVColl& McInfo)
{

  //Loop On the Collection and see if your tau jet is matched to one there
 //Also find the nearest Matched MC Particle to your Jet (to be complete)
 
 bool matched=false;
 double delta_min=100.;
 double mceta=0;
 double mcet=0;
 
 double matchDr=0.3;

 if(McInfo.size()>0)
  for(std::vector<LV>::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
   {
     	  double delta = ROOT::Math::VectorUtil::DeltaR(jet.p4().Vect(),*it);
	  if(delta<matchDr)
	    {
	      matched=true;
	      if(delta<delta_min)
		{
		  delta_min=delta;
		  mceta=it->eta();
		  mcet=it->Et();
		}
	    }
   }

  //Create Struct and send it out!
  MatchElementL25 match;
  match.matched=matched;
  match.deltar=delta_min;
  match.mcEta = mceta;
  match.mcEt = mcet;

 return match;
}



void L25TauAnalyzer::beginJob() {
}


void L25TauAnalyzer::endJob() {
   l25file->Write();
}


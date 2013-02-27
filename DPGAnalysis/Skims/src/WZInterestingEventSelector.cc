// -*- C++ -*-
//
//
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "TLorentzVector.h"
//
// class declaration
//

using namespace reco;

class WZInterestingEventSelector : public edm::EDFilter {
public:
  struct event
  {
    long run;
    long event;
    long ls;
    int nEle;
    float maxPt;
    float maxPtEleEta;
    float maxPtElePhi;
    float invMass;
    float met;
    float metPhi;
  };

  explicit WZInterestingEventSelector(const edm::ParameterSet&);
  ~WZInterestingEventSelector();
  
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob(); 
  bool electronSelection( const GsfElectron* eleRef , math::XYZPoint bspotPosition);  
  // ----------member data ---------------------------

  //std::vector<event> interestingEvents_;  

  //Pt cut
  float ptCut_;
  int missHitCut_;

  //EB ID+ISO cuts
  float eb_trIsoCut_;
  float eb_ecalIsoCut_;
  float eb_hcalIsoCut_;
  float eb_hoeCut_;
  float eb_seeCut_;

  //EE ID+ISO cuts
  float ee_trIsoCut_;
  float ee_ecalIsoCut_;
  float ee_hcalIsoCut_;
  float ee_hoeCut_;
  float ee_seeCut_;

  //met Cut
  float metCut_;

  //invMass Cut
  float invMassCut_;

  edm::InputTag electronCollection_;
  edm::InputTag pfMetCollection_;
  edm::InputTag offlineBSCollection_;

};

//
// constructors and destructor
//
WZInterestingEventSelector::WZInterestingEventSelector(const edm::ParameterSet& iConfig)
{
  ptCut_   = iConfig.getParameter<double>("ptCut");
  missHitCut_ = iConfig.getParameter<int>("missHitsCut");
  
  eb_trIsoCut_   = iConfig.getParameter<double>("eb_trIsoCut");
  eb_ecalIsoCut_   = iConfig.getParameter<double>("eb_ecalIsoCut");
  eb_hcalIsoCut_   = iConfig.getParameter<double>("eb_hcalIsoCut");
  eb_hoeCut_   = iConfig.getParameter<double>("eb_hoeCut");
  eb_seeCut_   = iConfig.getParameter<double>("eb_seeCut");

  ee_trIsoCut_   = iConfig.getParameter<double>("ee_trIsoCut");
  ee_ecalIsoCut_   = iConfig.getParameter<double>("ee_ecalIsoCut");
  ee_hcalIsoCut_   = iConfig.getParameter<double>("ee_hcalIsoCut");
  ee_hoeCut_   = iConfig.getParameter<double>("ee_hoeCut");
  ee_seeCut_   = iConfig.getParameter<double>("ee_seeCut");
  
  metCut_ = iConfig.getParameter<double>("metCut");
  invMassCut_ = iConfig.getParameter<double>("invMassCut");

  electronCollection_ = iConfig.getUntrackedParameter<edm::InputTag>("electronCollection",edm::InputTag("gsfElectrons"));
  pfMetCollection_ = iConfig.getUntrackedParameter<edm::InputTag>("pfMetCollection",edm::InputTag("pfMet"));
  offlineBSCollection_ = iConfig.getUntrackedParameter<edm::InputTag>("offlineBSCollection",edm::InputTag("offlineBeamSpot"));
  
}


WZInterestingEventSelector::~WZInterestingEventSelector()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------

void WZInterestingEventSelector::endJob() { 

//   if (interestingEvents_.size()<1)
//     return;

//   std::ostringstream oss;
//   for (unsigned int iEvent=0;iEvent<interestingEvents_.size();++iEvent)
//     {
//       oss << "==================================" << std::endl;
//       oss << "Run: " << interestingEvents_[iEvent].run << " Event: " <<  interestingEvents_[iEvent].event << " LS: " <<  interestingEvents_[iEvent].ls << std::endl;
//       oss << "nGoodEle: " << interestingEvents_[iEvent].nEle << " maxPt " << interestingEvents_[iEvent].maxPt <<  " maxPtEta " << interestingEvents_[iEvent].maxPtEleEta << " maxPtPhi " << interestingEvents_[iEvent].maxPtElePhi << std::endl;
//       oss << "invMass " << interestingEvents_[iEvent].invMass << " met " << interestingEvents_[iEvent].met << " metPhi " << interestingEvents_[iEvent].metPhi << std::endl;
//     }
//   std::string mailText;
//   mailText = oss.str();

//   std::ofstream outputTxt;
//   outputTxt.open("interestingEvents.txt");
//   outputTxt << mailText;
//   outputTxt.close();

  //Sending email
//   std::ostringstream subject;
//   subject << "Interesting events in Run#" << interestingEvents_[0].run; 

//   std::ostringstream command;
//   command << "cat interestingEvents.txt | mail -s \"" << subject.str() << "\" Paolo.Meridiani@cern.ch";
  
//   std::string commandStr = command.str();
//   char* pch = (char*)malloc( sizeof( char ) *(commandStr.length() +1) );
//   string::traits_type::copy( pch, commandStr.c_str(), commandStr.length() +1 );
//   int i=system(pch);
  
}

bool WZInterestingEventSelector::electronSelection( const GsfElectron* eleRef , math::XYZPoint bspotPosition )
{


//   if (eleRef->trackerDrivenSeed() && !eleRef->ecalDrivenSeed()) 
//     return false;
  
//   if (eleRef->ecalDrivenSeed())
//     {

  if (eleRef->pt()<ptCut_) return false;

  if (eleRef->isEB())
    {
       if (eleRef->dr03TkSumPt()/eleRef->pt()>eb_trIsoCut_) return false;
       if (eleRef->dr03EcalRecHitSumEt()/eleRef->pt()>eb_ecalIsoCut_) return false;
       if (eleRef->dr03HcalTowerSumEt()/eleRef->pt()>eb_hcalIsoCut_) return false;
       if (eleRef->sigmaIetaIeta()>eb_seeCut_) return false;
       if (eleRef->hcalOverEcal()>eb_hoeCut_) return false;
    }
  else if (eleRef->isEE())
    {
      if (eleRef->dr03TkSumPt()/eleRef->pt()>ee_trIsoCut_) return false;
      if (eleRef->dr03EcalRecHitSumEt()/eleRef->pt()>ee_ecalIsoCut_) return false;
      if (eleRef->dr03HcalTowerSumEt()/eleRef->pt()>ee_hcalIsoCut_) return false;
      if (eleRef->sigmaIetaIeta()>ee_seeCut_) return false;
      if (eleRef->hcalOverEcal()>ee_hoeCut_) return false;
    }
  
  if (eleRef->gsfTrack()->trackerExpectedHitsInner().numberOfHits()>missHitCut_) return false;
  
  return true;
}

bool
WZInterestingEventSelector::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  // using namespace edm;
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel(electronCollection_,gsfElectrons);

//   edm::Handle<reco::CaloMETCollection> caloMET;
//   iEvent.getByLabel(edm::InputTag("met"), caloMET);  

  edm::Handle<reco::PFMETCollection> pfMET;
  iEvent.getByLabel(pfMetCollection_, pfMET);  

  edm::Handle<reco::BeamSpot> pBeamSpot;
  iEvent.getByLabel(offlineBSCollection_, pBeamSpot);

  const reco::BeamSpot *bspot = pBeamSpot.product();
  math::XYZPoint bspotPosition = bspot->position();

  std::vector<const GsfElectron*> goodElectrons;  
  float ptMax=-999.;
  const GsfElectron* ptMaxEle=0;
  for(reco::GsfElectronCollection::const_iterator myEle=gsfElectrons->begin();myEle!=gsfElectrons->end();++myEle)
    {
      //Apply a minimal isolated electron selection
      if (!electronSelection(&(*myEle),bspotPosition)) continue;
      goodElectrons.push_back(&(*myEle));
      if (myEle->pt() > ptMax )
	{
	  ptMax =  myEle->pt();
	  ptMaxEle = &(*myEle);
	}
    }

  float maxInv=-999.;
  TLorentzVector v1;
  if (ptMaxEle)
    v1.SetPtEtaPhiM(ptMaxEle->pt(),ptMaxEle->eta(),ptMaxEle->phi(),0);
  if (goodElectrons.size()>1)
    {
      for (unsigned int iEle=0; iEle<goodElectrons.size(); ++iEle)
	if (goodElectrons[iEle]!=ptMaxEle && (goodElectrons[iEle]->charge() * ptMaxEle->charge() == -1) )
	  {
	    TLorentzVector v2;
	    v2.SetPtEtaPhiM(goodElectrons[iEle]->pt(),goodElectrons[iEle]->eta(),goodElectrons[iEle]->phi(),0.);
	    if ( (v1+v2).M() > maxInv )
	      maxInv = (v1+v2).M();
	  }
    }

  //Z filt: Retain event if more then 1 good ele and invMass above threshold (zee)
  if (goodElectrons.size()>1 && maxInv > invMassCut_)
    {
      //interestingEvents_.push_back(thisEvent);
      return true;
    }

  //W filt: Retain event also event with at least 1 good ele and some met
  if (goodElectrons.size()>=1 &&  (pfMET->begin()->et()>metCut_))
    {
      //interestingEvents_.push_back(thisEvent);
      return true;
    }
  
  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(WZInterestingEventSelector);

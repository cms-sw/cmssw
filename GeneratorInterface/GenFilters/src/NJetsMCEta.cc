// -*- C++ -*-
//
// Package:   NJetsMC
// Class:     NJetsMC
// 
/**\class NJetsMC NJetsMC.cc

 Description: Filter for DPS MC generation.

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  "Nathaniel Odell"
//         Created:  Thu Aug 12 09:24:46 CDT 2010
// $Id: NJetsMC.cc,v 1.1 2011/03/23 14:46:46 mucib Exp $
// then moved to more general N-jets purpose in GeneratorInterface/GenFilters
// Modified by Qiang Li on Dec 2 2013 to add eta cuts for the leading 2 jets


// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h" 
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TROOT.h"
#include "TH1F.h"
#include "TFile.h"
#include "TSystem.h"
#include "TLorentzVector.h"
#include <iostream>


using namespace edm;
using namespace std;
using namespace reco;
 
struct sortPt
{
bool operator()(TLorentzVector* s1, TLorentzVector* s2) const
  {
    return s1->Pt() >= s2->Pt();
  }
} mysortPt;


//
// class declaration
//

class NJetsMCEta : public edm::EDFilter 
{
public:
  explicit NJetsMCEta(const edm::ParameterSet&);
  ~NJetsMCEta();
  
private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  // ----------member data ---------------------------
  
  edm::InputTag GenHandle_;
  Int_t  njets_;
  double minpt_;
  double maxeta_;
  double mineta_;

  vector<TLorentzVector*> *pjet;

};

NJetsMCEta::NJetsMCEta(const edm::ParameterSet& iConfig):
  GenHandle_(iConfig.getUntrackedParameter<InputTag>("GenTag")),
  njets_(iConfig.getParameter<int32_t>("Njets")),
  minpt_(iConfig.getParameter<double>("MinPt")),
  maxeta_(iConfig.getParameter<double>("MaxEta")),
  mineta_(iConfig.getParameter<double>("MinEta"))
{
}


NJetsMCEta::~NJetsMCEta()
{
}

bool NJetsMCEta::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   Handle<reco::GenJetCollection> GenJets;
   iEvent.getByLabel(GenHandle_, GenJets);
 
   vector<TLorentzVector*> jet;

   Int_t count = 0;
   bool result = false;

 

   for(GenJetCollection::const_iterator iJet = GenJets->begin(); iJet != GenJets->end(); ++iJet)
     {
       const reco::Candidate* myJet = &(*iJet); 
       TLorentzVector *dummy = new TLorentzVector(0,0,0,0);
       dummy->SetPtEtaPhiE(myJet->pt(),myJet->eta(),myJet->energy(),myJet->phi());
       jet.push_back(dummy);
     }

   pjet = &jet ;

   sort (pjet->begin(), pjet->end(), mysortPt);
 
   if(pjet->size()>0 && pjet->at(0)->Pt() > minpt_ && abs(pjet->at(0)->Eta()) < maxeta_ && abs(pjet->at(0)->Eta()) > mineta_) ++count;
   if(pjet->size()>1 && pjet->at(1)->Pt() > minpt_ && abs(pjet->at(1)->Eta()) < maxeta_ && abs(pjet->at(1)->Eta()) > mineta_) ++count;


   if( count >= njets_ )
      result = true;

   return result;
}

void NJetsMCEta::beginJob()
{
}

void NJetsMCEta::endJob()
{
}
 


//define this as a plug-in
DEFINE_FWK_MODULE(NJetsMCEta);

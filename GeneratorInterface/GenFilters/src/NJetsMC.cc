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
// then moved to more general N-jets purpose in GeneratorInterface/GenFilters
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TROOT.h"
#include "TH1F.h"
#include "TFile.h"
#include "TSystem.h"
#include <iostream>

using namespace edm;
using namespace std;
using namespace reco;

//
// class declaration
//

class NJetsMC : public edm::EDFilter 
{
public:
  explicit NJetsMC(const edm::ParameterSet&);
  ~NJetsMC();
  
private:
  virtual void beginJob() override ;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;
  
  // ----------member data ---------------------------
  
  edm::InputTag GenHandle_;
  Int_t  njets_;
  double minpt_;

};

NJetsMC::NJetsMC(const edm::ParameterSet& iConfig):
  GenHandle_(iConfig.getUntrackedParameter<InputTag>("GenTag")),
  njets_(iConfig.getParameter<int32_t>("Njets")),
  minpt_(iConfig.getParameter<double>("MinPt"))
{
}


NJetsMC::~NJetsMC()
{
}

bool NJetsMC::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   Handle<reco::GenJetCollection> GenJets;
   iEvent.getByLabel(GenHandle_, GenJets);

   Int_t count = 0;
   bool result = false;

   for(GenJetCollection::const_iterator iJet = GenJets->begin(); iJet != GenJets->end(); ++iJet)
     {
       reco::GenJet myJet = reco::GenJet(*iJet);

       if(myJet.pt() > minpt_) ++count;
     }

   if( count >= njets_ )
      result = true;

   return result;
}

void NJetsMC::beginJob()
{
}

void NJetsMC::endJob()
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(NJetsMC);

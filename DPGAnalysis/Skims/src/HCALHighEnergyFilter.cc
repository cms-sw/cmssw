// -*- C++ -*-
//
// Package:    HCALHighEnergyFilter
// Class:      HCALHighEnergyFilter
// 
/**\class HCALHighEnergyFilter HCALHighEnergyFilter.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kenneth Case Rossato
//         Created:  Tue Aug 19 16:13:10 CEST 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include <string>

//
// class declaration
//

using namespace edm;
using namespace l1extra;

class HCALHighEnergyFilter : public edm::EDFilter {
   public:
      explicit HCALHighEnergyFilter(const edm::ParameterSet&);
      ~HCALHighEnergyFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
  bool jetGood(L1JetParticleCollection::const_iterator &);
      // ----------member data ---------------------------

  //edm::InputTag inputTag_;
  //  std::string HLTPath_;

  edm::InputTag centralTag, tauTag;
  double jet_threshold;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HCALHighEnergyFilter::HCALHighEnergyFilter(const edm::ParameterSet& iConfig)
  : /*inputTag_ (iConfig.getParameter<edm::InputTag>("TriggerResultsTag")),
      HLTPath_(iConfig.getParameter<std::string>("HLTPath"))*/
  centralTag(iConfig.getUntrackedParameter<edm::InputTag>("CentralJets")),
  tauTag(iConfig.getUntrackedParameter<edm::InputTag>("TauJets")),
  jet_threshold(iConfig.getUntrackedParameter("JetThreshold", 0.))
{
   //now do what ever initialization is needed

}


HCALHighEnergyFilter::~HCALHighEnergyFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

bool
HCALHighEnergyFilter::jetGood(L1JetParticleCollection::const_iterator &cit) {
  return cit->energy() >= jet_threshold;
}

// ------------ method called on each new Event  ------------
bool
HCALHighEnergyFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   /* Do this with standard HLTHighLevel filter
   Handle<TriggerResults> trh;
   iEvent.getByLabel(inputTag_, trh);
   //iEvent.getByLabel("TriggerResults", "HLT", trh);

   // let the attempted indirection throw a more descriptive exception for us
   //if (!trh.isValid()) throw cms::Exception("Event content") << " Couldn't retrieve valid trigger results.\n";

   TriggerNames triggerNames_;
   triggerNames_.init(*trh);

   unsigned int pathByIndex = triggerNames_.triggerIndex(HLTPath_);

   if (pathByIndex >= trh->size()) throw cms::Exception("Configuration")
     << " Unknown HLT path name\n";

   return trh->accept(pathByIndex);
   */

   Handle<L1JetParticleCollection> JetsCentral;
   iEvent.getByLabel(centralTag,JetsCentral);
   //   iEvent.getByLabel("hltL1extraParticles", "Central",JetsCentral);

   // not interested
   //Handle<L1JetParticleCollection> JetsForward;
   //iEvent.getByLabel(BranchTag_,"Forward",JetsForward);

   Handle<L1JetParticleCollection> JetsTau;
   iEvent.getByLabel(tauTag,JetsTau);
   //   iEvent.getByLabel("hltL1extraParticles", "Tau",JetsTau);

   for (L1JetParticleCollection::const_iterator cit = JetsCentral->begin();
	cit != JetsCentral->end(); cit++) {
     if (jetGood(cit)) return true;
   }

   for (L1JetParticleCollection::const_iterator cit = JetsTau->begin();
	cit != JetsTau->end(); cit++) {
     if (jetGood(cit)) return true;
   }

   return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HCALHighEnergyFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HCALHighEnergyFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HCALHighEnergyFilter);

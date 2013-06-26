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
// $Id: HCALHighEnergyFilter.cc,v 1.5 2013/02/27 20:17:14 wmtan Exp $
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

#include "DataFormats/Common/interface/TriggerResults.h"
//#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
//#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <string>

//
// class declaration
//

using namespace edm;
//using namespace l1extra;

class HCALHighEnergyFilter : public edm::EDFilter {
   public:
      explicit HCALHighEnergyFilter(const edm::ParameterSet&);
      ~HCALHighEnergyFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
  //  bool jetGood(L1JetParticleCollection::const_iterator &);
  bool jetGood(reco::CaloJetCollection::const_iterator &);
      // ----------member data ---------------------------

  //  edm::InputTag centralTag, tauTag;
  edm::InputTag jet_tag;
  double jet_threshold;
  double eta_cut;
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
  :
  //  centralTag(iConfig.getUntrackedParameter<edm::InputTag>("CentralJets")),
  //  tauTag(iConfig.getUntrackedParameter<edm::InputTag>("TauJets")),
  jet_tag(iConfig.getParameter<edm::InputTag>("JetTag")),
  jet_threshold(iConfig.getParameter<double>("JetThreshold")),
  eta_cut(iConfig.getParameter<double>("EtaCut"))
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
//HCALHighEnergyFilter::jetGood(L1JetParticleCollection::const_iterator &cit) {
HCALHighEnergyFilter::jetGood(reco::CaloJetCollection::const_iterator &cit) {
  if (cit->energy() >= jet_threshold && std::fabs(cit->eta()) <= eta_cut)
    return true;
  return false;
}

// ------------ method called on each new Event  ------------
bool
HCALHighEnergyFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   //Handle<L1JetParticleCollection> JetsCentral;
   //iEvent.getByLabel(centralTag,JetsCentral);

   //Handle<L1JetParticleCollection> JetsTau;
   //iEvent.getByLabel(tauTag,JetsTau);

   Handle<reco::CaloJetCollection> Jets;
   iEvent.getByLabel(jet_tag, Jets);

   /*
   for (L1JetParticleCollection::const_iterator cit = JetsCentral->begin();
	cit != JetsCentral->end(); cit++) {
     if (jetGood(cit)) return true;
   }

   for (L1JetParticleCollection::const_iterator cit = JetsTau->begin();
	cit != JetsTau->end(); cit++) {
     if (jetGood(cit)) return true;
   }
   */

   for (reco::CaloJetCollection::const_iterator cit = Jets->begin();
	cit != Jets->end(); cit++) {
     if (jetGood(cit)) return true;
   }

   return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HCALHighEnergyFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HCALHighEnergyFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HCALHighEnergyFilter);

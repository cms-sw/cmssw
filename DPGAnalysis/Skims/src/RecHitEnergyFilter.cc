// -*- C++ -*-
//
// Package:    RecHitEnergyFilter
// Class:      RecHitEnergyFilter
// 
/**\class RecHitEnergyFilter RecHitEnergyFilter.cc JacksonJ/RecHitEnergyFilter/src/RecHitEnergyFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  James Jackson
//         Created:  Mon Jan 11 09:57:58 CET 2010
// $Id: RecHitEnergyFilter.cc,v 1.2 2013/02/27 20:17:14 wmtan Exp $
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
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//
// class declaration
//

class RecHitEnergyFilter : public edm::EDFilter {
   public:
      explicit RecHitEnergyFilter(const edm::ParameterSet&);
      ~RecHitEnergyFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      // RecHit input tags
      edm::InputTag ebRecHitsTag_;
      edm::InputTag eeRecHitsTag_;
      
      // RecHit handles
      edm::Handle<EcalRecHitCollection> ebRecHitsH_;
      edm::Handle<EcalRecHitCollection> eeRecHitsH_;
      const EcalRecHitCollection *ebRecHits_;
      const EcalRecHitCollection *eeRecHits_;     

      // Procesing control
      bool doEb_;
      bool doEe_;
      double ebThresh_;
      double eeThresh_;
};

//
// constructors and destructor
//
RecHitEnergyFilter::RecHitEnergyFilter(const edm::ParameterSet& iConfig)
{
   doEb_ = iConfig.getParameter<bool>("DoEB");
   doEe_ = iConfig.getParameter<bool>("DoEE");
   ebRecHitsTag_ = iConfig.getParameter<edm::InputTag>("EBRecHits");
   eeRecHitsTag_ = iConfig.getParameter<edm::InputTag>("EERecHits");
   ebThresh_ = iConfig.getParameter<double>("EBThresh");
   eeThresh_ = iConfig.getParameter<double>("EEThresh");
}


RecHitEnergyFilter::~RecHitEnergyFilter()
{
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
RecHitEnergyFilter::filter(edm::Event& evt, const edm::EventSetup& iSetup)
{
   if(doEb_)
   {
      evt.getByLabel(ebRecHitsTag_, ebRecHitsH_);
      ebRecHits_ = ebRecHitsH_.product();
      for(EcalRecHitCollection::const_iterator it = ebRecHits_->begin(); it != ebRecHits_->end(); ++it)
      {
         double hitE = it->energy();
         if(hitE > ebThresh_)
         {
            return true;
         }
      }
   }
   if(doEe_)
   {
      evt.getByLabel(eeRecHitsTag_, eeRecHitsH_);
      eeRecHits_ = eeRecHitsH_.product();
      for(EcalRecHitCollection::const_iterator it = eeRecHits_->begin(); it != eeRecHits_->end(); ++it)
      {
         double hitE = it->energy();
         if(hitE > eeThresh_)
         {
            return true;
         }
      }
   }

   return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
RecHitEnergyFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RecHitEnergyFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(RecHitEnergyFilter);

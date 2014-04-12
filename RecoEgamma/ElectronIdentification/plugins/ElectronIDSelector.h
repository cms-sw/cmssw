#ifndef ELECTRONIDSELECTOR
#define ELECTRONIDSELECTOR

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

template<class algo>
struct ElectronIDSelector{
 public:
   explicit ElectronIDSelector(const edm::ParameterSet& iConfig, edm::ConsumesCollector && iC) :
            select_(iConfig, iC),
	    threshold_(iConfig.getParameter<double>("threshold"))
   {
   }

   virtual ~ElectronIDSelector() {};

   // Collections to be selected
   typedef reco::GsfElectronCollection collection;
   typedef std::vector<reco::GsfElectronRef> container ;
   //typedef std::vector<reco::GsfElectron> container ;
   typedef container::const_iterator const_iterator;

   //define iterators with above typedef
   const_iterator begin () const { return selected_.begin () ; }
   const_iterator end () const { return  selected_.end () ; }

   void select(const edm::Handle<reco::GsfElectronCollection>& _electrons,
               const edm::Event& iEvent ,
	       const edm::EventSetup& iEs)
   {
     edm::Handle<reco::GsfElectronCollection> electrons = _electrons;
     selected_.clear();
     select_.newEvent(iEvent, iEs);
     // Loop over electrons
     unsigned int i = 0 ;
     for ( reco::GsfElectronCollection::const_iterator eleIt = electrons->begin () ;
      	 			                       eleIt != electrons->end () ;
   						       ++eleIt )
	{
	 edm::Ref<reco::GsfElectronCollection> electronRef(electrons,i);
	 if (select_((*eleIt),iEvent,iEs) > threshold_)
	     selected_.push_back (electronRef) ;
	     //selected_.push_back ( & * eleIt) ;
	 ++i;
	}
   }

 private:
   container selected_ ;
   algo select_ ;
   double threshold_ ;

};

#endif

#ifndef PhysicsTools_UtilAlgos_interface_FWLiteFilterWrapper_h
#define PhysicsTools_UtilAlgos_interface_FWLiteFilterWrapper_h

/**
  \class    FWLiteFilterWrapper FWLiteFilterWrapper.h "PhysicsTools/UtilAlgos/interface/FWLiteFilterWrapper.h"
  \brief    Implements a wrapper around an FWLite-friendly selector to "convert" it into a full EDFilter


  \author Salvatore Rappoccio
  \version  $Id: FWLiteFilterWrapper.h,v 1.1 2010/07/22 16:43:55 srappocc Exp $
*/


#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/EventBase.h"

#include <boost/shared_ptr.hpp>

namespace edm {

template<class T>
class FWLiteFilterWrapper : public EDFilter {

 public:

  /// Pass the parameters to filter_
 FWLiteFilterWrapper(const edm::ParameterSet& pset)
  {
    filter_  = boost::shared_ptr<T>( new T(pset) );
  }

  /// Destructor does nothing
  virtual ~FWLiteFilterWrapper() {}
 

  /// Pass the event to the filter. NOTE! We can't use the eventSetup in FWLite so ignore it.
  virtual bool filter( edm::Event & event, const edm::EventSetup& eventSetup)
  {
    edm::EventBase & ev = event;
    return (*filter_)(ev);
  }


 protected:
  boost::shared_ptr<T> filter_;
};

}

#endif

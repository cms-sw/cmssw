#ifndef PhysicsTools_UtilAlgos_interface_FWLiteFilterWrapper_h
#define PhysicsTools_UtilAlgos_interface_FWLiteFilterWrapper_h

#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/**
  \class    FWLiteFilterWrapper FWLiteFilterWrapper.h "PhysicsTools/UtilAlgos/interface/FWLiteFilterWrapper.h"
  \brief    Implements a wrapper around an FWLite-friendly selector to "convert" it into a full EDFilter

  Please Note: THIS FILE HAS BEEN DEPRECATED. IT HAS BEEN MOVED TO 
               PhysicsTools/UtilsAlgos/interface/EDFilterWrapper.h

  \author Salvatore Rappoccio
  \version  $Id: FWLiteFilterWrapper.h,v 1.2 2010/10/14 09:53:48 snaumann Exp $
*/

namespace edm {

  template <class T>
  class FWLiteFilterWrapper : public EDFilter {
  public:
    /// Pass the parameters to filter_
    FWLiteFilterWrapper(const edm::ParameterSet& pset) {
      edm::LogWarning("FWLiteFilterWrapper") << "Please Note: THIS FILE HAS BEEN DEPRECATED. IT HAS BEEN MOVED TO \n"
                                             << "PhysicsTools/UtilsAlgos/interface/EDFilterWrapper.h";

      filter_ = std::shared_ptr<T>(new T(pset));
    }

    /// Destructor does nothing
    virtual ~FWLiteFilterWrapper() {}

    /// Pass the event to the filter. NOTE! We can't use the eventSetup in FWLite so ignore it.
    virtual bool filter(edm::Event& event, const edm::EventSetup& eventSetup) {
      edm::EventBase& ev = event;
      return (*filter_)(ev);
    }

  protected:
    std::shared_ptr<T> filter_;
  };

}  // namespace edm

#endif

#ifndef PhysicsTools_UtilAlgos_interface_EDFilterWrapper_h
#define PhysicsTools_UtilAlgos_interface_EDFilterWrapper_h

/**
  \class    EDFilterWrapper EDFilterWrapper.h "PhysicsTools/UtilAlgos/interface/EDFilterWrapper.h"
  \brief    Wrapper class for a class of type BasicFilter to "convert" it into a full EDFilter

   This template class is a wrapper around classes of type BasicFilter as defined in the
   BasicFilter.h file of this package. From this class the wrapper expects the following
   member functions:

   + a constructor with a const edm::ParameterSet& as input.
   + a filter function with an const edm::EventBase& as input

   the function is called within the wrapper. The wrapper translates the common class into
   a basic EDFilter as shown below:

   #include "PhysicsTools/UtilAlgos/interface/EDFilterWrapper.h"
   #include "PhysicsTools/SelectorUtils/interface/PVSelector.h"

   typedef edm::FilterWrapper<PVSelector> PrimaryVertexFilter;

   #include "FWCore/Framework/interface/MakerMacros.h"
   DEFINE_FWK_MODULE(PrimaryVertexFilter);

   You can find this example in the plugins directory of this package. With this wrapper class
   we have the use case in mind that you keep classes, which easily can be used both within the
   full framework and within FWLite.

   NOTE: in the current implementation this wrapper class does not support use of the EventSetup.
   If you want to make use of this feature we recommend you to start from an EDFilter from the
   very beginning and just to stay within the full framework.
*/


#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/Event.h"
#include <boost/shared_ptr.hpp>

namespace edm {

  template<class T>
  class FilterWrapper : public EDFilter {

  public:
    /// default contructor
    FilterWrapper(const edm::ParameterSet& cfg){ filter_ = boost::shared_ptr<T>( new T(cfg, consumesCollector()) ); }
    /// default destructor
    virtual ~FilterWrapper(){}
    /// everything which has to be done during the event loop. NOTE: We can't use the eventSetup in FWLite so ignore it
    virtual bool filter(edm::Event& event, const edm::EventSetup& eventSetup){
      edm::EventBase & eventBase = dynamic_cast<edm::EventBase &>(event);
      edm::EventBase const & eventBaseConst = const_cast<edm::EventBase const &>(eventBase);
      return (*filter_)(eventBaseConst);
    }

  protected:
    /// shared pointer to analysis class of type BasicAnalyzer
    boost::shared_ptr<T> filter_;
  };

}

#endif

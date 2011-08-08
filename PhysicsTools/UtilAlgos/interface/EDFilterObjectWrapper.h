#ifndef PhysicsTools_UtilAlgos_interface_EDFilterObjectWrapper_h
#define PhysicsTools_UtilAlgos_interface_EDFilterObjectWrapper_h

/**
  \class    FilterWrapper FilterWrapper.h "PhysicsTools/UtilAlgos/interface/FilterWrapper.h"
  \brief    Wrapper class around a class of type BasicAnalyzer to "convert" it into a full EDFilter

   This template class is a wrapper round classes of type Selector<T> and similar signature. 


   This operates on container classes of type C which roughly satisfy std::vector template
   parameters. 

   From this class the wrapper expects the following 
   member functions:
   
   + a contructor with a const edm::ParameterSet& as input.
   + a filter function that operates on classes of type C::value_type

   
   the function is called within the wrapper. The wrapper translates the common class into 
   a basic EDFilter as shown below: 
   
   #include "PhysicsTools/UtilAlgos/interface/EDFilterObjectWrapper.h"
   #include "PhysicsTools/SelectorUtils/interface/PFJetIdSelectionFunctor.h"
   
   typedef edm::FilterWrapper<PFJetIdSelectionFunctor> PFJetIdFilter;
   
   #include "FWCore/Framework/interface/MakerMacros.h"
   DEFINE_FWK_MODULE(PFJetIdFilter);
   
   You can find this example in the plugins directory of this package. 

   NOTE: in the current implementation this wrapper class does not support use of the EventSetup. 
   If you want to make use of this feature we recommend you to start from an EDAnalyzer from the 
   very beginning and just to stay within the full framework.
*/


#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/shared_ptr.hpp>

namespace edm {

  template<class T, class C>
  class FilterObjectWrapper : public EDFilter {
    
  public:

    /// Some convenient typedefs. Recall that C is a container class.
    typename C::iterator        iterator;
    typename C::const_iterator  const_iterator;

    /// default contructor.
    /// Declares the output (type "C") and the filter (of type T, operates on C::value_type)
    FilterObjectWrapper(const edm::ParameterSet& cfg) :
      src_( cfg.getParameter<edm::InputTag>("src"))
    { 
      filter_ = boost::shared_ptr<T>( new T(cfg.getParameter<edm::ParameterSet>("filterParams")) );
      if ( cfg.exists("filter") ) {
	doFilter_ = cfg.getParameter<bool>("filter");
      } else {
	doFilter_ = false;
      }
      produces<C>();
    }
    /// default destructor
    virtual ~FilterObjectWrapper(){}
    /// everything which has to be done during the event loop. NOTE: We can't use the eventSetup in FWLite so ignore it
    virtual bool filter(edm::Event& event, const edm::EventSetup& eventSetup){ 
      
      // Create a collection of the objects to put
      std::auto_ptr<C> objsToPut( new C() );
      // Get the handle to the objects in the event.
      edm::Handle<C> h_c;     
      event.getByLabel( src_, h_c );

      // Loop through and add passing value_types to the output vector
      for ( typename C::const_iterator ibegin = h_c->begin(), iend = h_c->end(), i = ibegin;
	    i != iend; ++i ) {
	if ( (*filter_)(*i) ) {
	  objsToPut->push_back( *i );
	}
      }

      // put objs in Event
      bool pass = objsToPut->size() > 0;
      event.put(objsToPut);
      // Return
      if ( doFilter_ )
	return pass;
      else
	return true;
    }
    
  protected:
    /// InputTag of the input source
    edm::InputTag        src_;
    /// shared pointer to analysis class of type BasicAnalyzer
    boost::shared_ptr<T> filter_;
    /// whether or not to filter based on size
    bool                 doFilter_;
  };

}

#endif

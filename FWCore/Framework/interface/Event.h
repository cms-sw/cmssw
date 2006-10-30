#ifndef Framework_Event_h
#define Framework_Event_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     Event
// 
/**\class Event Event.h FWCore/Framework/interface/Event.h

Description: This is the primary interface for accessing EDProducts from a single collision and inserting new derived products.

Usage:

Getting Data

The edm::Event class provides many 'get*" methods for getting data it contains.  

The primary method for getting data is to use getByLabel(). The labels are the label of the module assigned
in the configuration file and the 'product instance label' (which can be omitted in the case the 'product instance label'
is the default value).  The C++ type of the event product plus the two labels uniquely identify a product in the Event.

\code
  edm::Handle<AppleCollection> apples;
  event.getByLabel("tree",apples);
\endcode

\code
  edm::Handle<FruitCollection> fruits;
  event.getByLabel("market", "apple", fruits);
\endcode


Putting Data

\code
  std::auto_ptr<AppleCollection> pApples( new AppleCollection );
  
  //fill the collection
  ...
  event.put(pApples);
\endcode

\code
  std::auto_ptr<FruitCollection> pFruits( new FruitCollection );

  //fill the collection
  ...
  event.put("apple", pFruits);
\endcode


Getting a reference to an event product before that product is put into the event
NOTE: The edm::RefProd returned will not work until after the edm::Event has 
been committed (which happens after the EDProducer::produce method has ended)
\code
  std::auto_ptr<AppleCollection> pApples( new AppleCollection);

  edm::RefProd<AppleCollection> refApples = event.getRefBeforePut<AppleCollection>();

  //do loop and fill collection
  for( unsigned int index = 0; ..... ) {
    ....
    apples->push_back( Apple(...) );
  
    //create an edm::Ref to the new object
    edm::Ref<AppleCollection> ref(refApples, index);
    ....
  }
\endcode

*/
/*----------------------------------------------------------------------

$Id: Event.h,v 1.45 2006/10/27 20:56:42 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EventAux.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"

#include "FWCore/Framework/interface/DataViewImpl.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

#if GCC_PREREQUISITE(3,4,4)
  class Event : private DataViewImpl
#else
  // Bug in gcc3.2.3 compiler forces public inheritance
  class Event : public DataViewImpl
#endif
  {
  public:
    Event(EventPrincipal& dbk, const ModuleDescription& md);
    virtual ~Event();

    // AUX functions.
    EventID id() const {return aux_.id();}
    Timestamp time() const {return aux_.time();}

    using DataViewImpl::get;
    using DataViewImpl::getAllProvenance;
    using DataViewImpl::getByLabel;
    using DataViewImpl::getByType;
    using DataViewImpl::getMany;
    using DataViewImpl::getManyByType;
    using DataViewImpl::getProvenance;
    using DataViewImpl::put;

  private:
    // commit_() is called to complete the transaction represented by
    // this DataViewImpl. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class RawInputSource;
    friend class FilterWorker;
    friend class ProducerWorker;

    EventAux const& aux_;
  };
}
#endif

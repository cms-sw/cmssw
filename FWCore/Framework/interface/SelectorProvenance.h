#ifndef Framework_SelectorProvenance_h
#define Framework_SelectorProvenance_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     SelectorProvenance
// 
/**\class SelectorProvenance SelectorProvenance.h FWCore/Framework/interface/SelectorProvenance.h

 Description: Helper interface used to access Provenance

 Usage:
    Provides access to information in the Provenance that is used by
    Selector's.  It does not allow access to those parts of the Provenance
    that should not be used by a Selector.  This restriction
    enforces limits on what a Selector can use and is the primary
    purpose this class exists.  There are parts of the Provenance
    that change from one event to the next and are not always
    filled when a Selector is used on events where unscheduled
    (aka OnDemand) execution is enabled.

    Simply for lack of time, I only implemented the portions of
    the provenance currently in use in Selector's.  This class should
    be extended to include all the event independent provenance
    (the part of the provenance available before the EDProduct is
    actually produced).  This should be easy and not involve anything
    more than adding other accessors and testing that they work properly.
    On the other hand, accessors for the event dependent provenance
    should probably never be added, at least not without a lot of
    difficult redesign and thought.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  15 March 2007
// $Id$
//

#include "DataFormats/Provenance/interface/Provenance.h"

namespace edm {
  class SelectorProvenance
  {

  public:

    SelectorProvenance(const Provenance& prov) : prov_(&prov) { }

    std::string const& className() const {return prov_->className();}
    std::string const& friendlyClassName() const {return prov_->friendlyClassName();}
    std::string const& moduleLabel() const {return prov_->moduleLabel();}
    std::string const& moduleName() const {return prov_->moduleName();}
    std::string const& processName() const {return prov_->processName();}
    edm::ProductID const& productID() const {return prov_->productID();}
    std::string const& productInstanceName() const {return prov_->productInstanceName();}

  private:

    const Provenance* prov_;
  };
}
#endif

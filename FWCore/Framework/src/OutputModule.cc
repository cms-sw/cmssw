/*----------------------------------------------------------------------
$Id: OutputModule.cc,v 1.8 2006/01/05 22:40:26 paterno Exp $
----------------------------------------------------------------------*/

#include <vector>

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace
{
  // This grotesque little function exists just to allow calling of
  // ConstProductRegistry::allBranchDescriptions in the context of
  // OutputModule's initialization list, rather than in the body of
  // the constructor.

  std::vector<edm::BranchDescription const*>
  getAllBranchDescriptions()
  {
    edm::Service<edm::ConstProductRegistry> reg;
    return reg->allBranchDescriptions();
  }
}

namespace edm {
  OutputModule::OutputModule(ParameterSet const& pset) : 
    nextID_(), 
    descVec_(),
    groupSelector_(pset,
		   getAllBranchDescriptions())
  {
    Service<ConstProductRegistry> reg;
    nextID_ = reg->nextID();

    // TODO: See if we can collapse descVec_ and groupSelector_ into a
    // single object. See the notes in the header for GroupSelector
    // for more information.

    ProductRegistry::ProductList::const_iterator it  = 
      reg->productList().begin();
    ProductRegistry::ProductList::const_iterator end = 
      reg->productList().end();

    for ( ; it != end; ++it)
      {
	if (selected(it->second)) descVec_.push_back(&it->second);
      }
  }

  OutputModule::~OutputModule() { }

  void OutputModule::beginJob(EventSetup const&) { }

  void OutputModule::endJob() { }

  bool OutputModule::selected(BranchDescription const& desc) const
  {
    return groupSelector_.selected(desc);
  }

  unsigned long OutputModule::nextID() const 
  {
    return nextID_;
  }
}

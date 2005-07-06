#include "FWCore/CoreFramework/interface/Provenance.h"

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.2 2005/06/07 22:42:45 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm
{
  Provenance::Provenance() :
    module(),
    product_id(),
    parents(),
    cid(),
    full_product_type_name(),
    friendly_product_type_name(),
    product_instance_name(),
    status(Success)
  { }

  Provenance::Provenance(const ModuleDescription& m) :
    module(m),
    product_id(),
    parents(),
    cid(),
    full_product_type_name(),
    friendly_product_type_name(),
    product_instance_name(),
    status(Success)
  { }

  void
  Provenance::write(std::ostream& os) const
  {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    os << "Provenance for: " << cid;
  }

    
  bool operator==(const Provenance& a, const Provenance& b)
  {
    return
      a.module == b.module 
      && a.cid == b.cid
      && a.full_product_type_name == b.full_product_type_name
      && a.product_instance_name == b.product_instance_name
      && a.status == b.status
      && a.parents == b.parents;
  }

}


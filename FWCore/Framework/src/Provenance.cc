#include "FWCore/CoreFramework/interface/Provenance.h"

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.3 2005/03/25 16:59:14 paterno Exp $

----------------------------------------------------------------------*/

namespace edm
{
  Provenance::Provenance() :
    module(),
    product_id(),
    parents(),
    cid(),
    friendly_product_type_name(),
    status(Success)
  { }

  Provenance::Provenance(const ModuleDescription& m) :
    module(m),
    product_id(),
    parents(),
    cid(),
    friendly_product_type_name(),
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
      && a.friendly_product_type_name == b.friendly_product_type_name
      && a.status == b.status
      && a.parents == b.parents;
  }

}


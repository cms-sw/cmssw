//////////////////////////////////////////////////////////////////////
//
// $Id: FileCatalog.cc,v 1.6 2007/09/05 21:11:24 wmtan Exp $
//
// Original Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Catalog/interface/FileCatalog.h"

namespace edm {

  FileCatalog::FileCatalog(PoolCatalog & poolcat) :
      catalog_(poolcat.catalog_),
      url_(),
      active_(false) {
  }

  FileCatalog::~FileCatalog() {
    if (active_) catalog_.commit();
    catalog_.disconnect();
  }

  void FileCatalog::commitCatalog() {
    catalog_.commit();
    catalog_.start();
  }

}

//////////////////////////////////////////////////////////////////////
//
// $Id: FileCatalog.cc,v 1.5 2007/08/06 19:53:06 wmtan Exp $
//
// Original Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Catalog/interface/FileCatalog.h"

namespace edm {

  FileCatalog::FileCatalog() :
      catalog_(),
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

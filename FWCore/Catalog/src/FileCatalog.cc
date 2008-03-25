//////////////////////////////////////////////////////////////////////
//
// $Id: FileCatalog.cc,v 1.4 2007/06/29 03:43:19 wmtan Exp $
//
// Original Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Catalog/interface/FileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/algorithm/string.hpp>

namespace edm {

  FileCatalog::FileCatalog(ParameterSet const& pset) :
      catalog_(),
      url_(pset.getUntrackedParameter<std::string>("catalog", std::string())),
      active_(false) {
    boost::trim(url_);
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

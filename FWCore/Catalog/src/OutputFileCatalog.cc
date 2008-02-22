//////////////////////////////////////////////////////////////////////
//
// $Id: OutputFileCatalog.cc,v 1.4 2007/06/29 03:43:19 wmtan Exp $
//
// Original Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Catalog/interface/OutputFileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/IFCAction.h"

#include <boost/algorithm/string.hpp>

namespace edm {
  OutputFileCatalog::OutputFileCatalog(ParameterSet const& pset) :
      FileCatalog(pset),
      fileName_(pset.getUntrackedParameter<std::string>("fileName")),
      logicalFileName_(pset.getUntrackedParameter<std::string>("logicalFileName", std::string())) {
    boost::trim(fileName_);
    if (fileName_.empty()) {
        throw edm::Exception(edm::errors::Configuration, "OutputFileCatalog::OutputFileCatalog()\n")
	  << "Empty 'fileName' parameter specified for output module.\n";
    }
    boost::trim(logicalFileName_);
    if (url().empty()) {
      url() = "file:PoolFileCatalog.xml"; // always for the output case
    } else {
      url() = toPhysical(url());
    }
    pool::URIParser parser(url());
    parser.parse();
    catalog().setWriteCatalog(parser.contactstring());
    catalog().connect();
    catalog().start();
    setActive();
  }

  OutputFileCatalog::~OutputFileCatalog() {}

  pool::FileCatalog::FileID OutputFileCatalog::registerFile(std::string const& pfn, std::string const& lfn) {
    pool::FileCatalog::FileID fid;
    {
      std::string type;
      pool::FCregister action;
      catalog().setAction(action);
      action.lookupFileByPFN(pfn, fid, type);
    }
    if (fid.empty()) {
      std::string fileType = "ROOT_Tree";
      pool::FCregister action;
      catalog().setAction(action);
      action.registerPFN(pfn, fileType, fid);
    }
    if (!lfn.empty()) {
      if (isPhysical(lfn)) {
        throw cms::Exception("IllegalCharacter") << "Logical file name '" << lfn
        << "' contains a colon (':'), which is illegal in an LFN.\n";
      }
      pool::FileCatalog::FileID fidl;
      {
        pool::FCregister action;
        catalog().setAction(action);
        action.lookupFileByLFN(lfn, fidl);
      }
      if (fidl.empty()) {
        pool::FCregister action;
        catalog().setAction(action);
        action.registerLFN(pfn, lfn);       
      }
    }
    return fid;
  }
}

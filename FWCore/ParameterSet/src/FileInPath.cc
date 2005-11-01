// ----------------------------------------------------------------------
// $Id: ParameterSet.cc,v 1.10 2005/10/27 18:27:17 wmtan Exp $
//
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "boost/filesystem/operations.hpp"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm
{
  FileInPath::FileInPath() :
    relativePath_(),
    canonicalFilename_(),
    isLocal_(false)
  { }

  FileInPath::FileInPath(const std::string& r) :
    relativePath_(r),
    canonicalFilename_(),
    isLocal_(false)
  {
    initialize_();
  }

  FileInPath::FileInPath(const char* r) :
    relativePath_( r ? r : throw Exception(edm::errors::FileInPathError) << "Relative path may not be null\n"),
    canonicalFilename_(),
    isLocal_(false)
  {
    initialize_();    
  }

  std::string const&
  FileInPath::relativePath() const
  {
    return relativePath_;
  }


  bool
  FileInPath::isLocal() const
  {
    return isLocal_;
  }

  std::string const&
  FileInPath::filename() const
  {
    return canonicalFilename_;
  }

  void
  FileInPath::write(std::ostream& os) const
  {
    os << relativePath_ << ' ' << isLocal_;    
  }


  void
  FileInPath::read(std::istream& is)
  {
    std::string relname;
    bool        local;
    is >> relname >> local;
    if (!is) return;
    relativePath_ = relname;
    isLocal_ = local;
  }

  //------------------------------------------------------------


  void 
  FileInPath::initialize_()
  {
    if (relativePath_.empty())
      throw Exception(edm::errors::FileInPathError)
	<< "Relative path may not be empty\n";

    // Find the file, under either "." or $CMSDATA.

    // Convert relative path to canonical form.

    // Convert the absolute path to canonical form.
  }

  
}





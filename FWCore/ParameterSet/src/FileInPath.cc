// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

// TODO: This file needs some clean-up, especially regarding the
// handling of environment variables. We can do better with
// translating them only once --- after we have settled down on how
// long the search path is allowed to be, and whether our only choices
// for the "official" directory is CMSSW_RELEASE_BASE or CMSSW_DATA_PATH.

#include <cstdlib>
#include <vector>
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Parse.h"

namespace bf = boost::filesystem;

namespace 
{
  /// These are the names of the environment variables which control
/// the behavior  of the FileInPath  class.  They are local to  this
/// class; other code should not even know about them!
    
  const std::string PathVariableName("CMSSW_SEARCH_PATH");
  // Environment variables for local and release areas: 
  const std::string LOCALTOP("CMSSW_BASE");
  const std::string RELEASETOP("CMSSW_RELEASE_BASE");
  const std::string DATATOP("CMSSW_DATA_PATH");

#if 1
  // Needed for backward compatibility prior to CMSSW_1_5_0_pre3.
  // String to serve as placeholder for release top. 
  // Do not change this value.
  const std::string BASE("BASE");
#endif
  const std::string version("V001");

  // Return false if the environment variable 'name is not found, and
  // true if it is found. If it is found, put the translation of the
  // environment variable into 'result'.
  bool envstring(std::string const& name,
		 std::string& result)
  {
    const char* val = getenv(name.c_str());
    if (val == 0) {
      return false;
    }
    result = val;
    return true;
  }


  // Check for existence of a file for the given relative path and
  // 'prefix'.
  // Return true if a file (not directory or symbolic link) is found
  // Return false is *nothing* is found
  // Throw an exception if either a directory or symbolic link is found.
  // If true is returned, then put the 
  bool locateFile(bf::path  p,
		  std::string const& relative)
  {
    p /= relative;

    if (!bf::exists(p)) return false;

    if (bf::is_directory(p))
      throw edm::Exception(edm::errors::FileInPathError)
	<< "Path " 
	<< p.native_directory_string()
	<< " is a directory, not a file\n";

    if (bf::symbolic_link_exists(p))
      throw edm::Exception(edm::errors::FileInPathError)
	<< "Path " 
	<< p.native_file_string()
	<< " is a symbolic link, not a file\n";

    return true;    
  }

}

namespace edm
{

  FileInPath::FileInPath() :
    relativePath_(),
    canonicalFilename_(),
    location_(Unknown)
  {
    getEnvironment();
  }

  FileInPath::FileInPath(const std::string& r) :
    relativePath_(r),
    canonicalFilename_(),
    location_(Unknown)
  {
    getEnvironment();
    initialize_();
  }

  FileInPath::FileInPath(const char* r) :
    relativePath_(r ?
		  r :
		  ((throw edm::Exception(edm::errors::FileInPathError)
		    << "Relative path must not be null\n"), r)),
    canonicalFilename_(),
    location_(Unknown)
  {
    getEnvironment();
    initialize_();    
  }

  FileInPath::FileInPath(FileInPath const& other) :
    relativePath_(other.relativePath_),
    canonicalFilename_(other.canonicalFilename_),
    location_(other.location_),
    localTop_(other.localTop_),
    releaseTop_(other.releaseTop_),
    dataTop_(other.dataTop_),
    searchPath_(other.searchPath_)
  {}

  FileInPath::~FileInPath() {}

  FileInPath&
  FileInPath::operator=(FileInPath const& other)
  {
    FileInPath temp(other);
    this->swap(temp);
    return *this;
  }

  void
  FileInPath::swap(FileInPath& other)
  {
    relativePath_.swap(other.relativePath_);
    canonicalFilename_.swap(other.canonicalFilename_);
    std::swap(location_, other.location_);
    localTop_.swap(other.localTop_);
    releaseTop_.swap(other.releaseTop_);
    dataTop_.swap(other.dataTop_);
    searchPath_.swap(other.searchPath_);
  }

  std::string
  FileInPath::relativePath() const
  {
    return relativePath_;
  }


  FileInPath::LocationCode
  FileInPath::location() const
  {
    return location_;
  }

  bool
  FileInPath::isLocal() const
  {
    return Local == location_;
  }

  std::string
  FileInPath::fullPath() const
  {
    return canonicalFilename_;
  }

  void
  FileInPath::write(std::ostream& os) const
  {
    if (location_ == Unknown) {
      os << version << ' ' << relativePath_ << ' ' << location_;
    } else if (location_ == Local) {
      // Guarantee a site independent value by stripping $LOCALTOP.
      if (localTop_.empty()) {
	throw edm::Exception(edm::errors::FileInPathError)
	  << "Environment Variable " 
	  << LOCALTOP
	  << " is not set.\n";
      }
      std::string::size_type pos = canonicalFilename_.find(localTop_);
      if (pos != 0) {
	throw edm::Exception(edm::errors::FileInPathError)
	  << "Path " 
	  << canonicalFilename_
	  << " is not in the local release area "
	  << localTop_
	  << "\n";
      }
      os << version << ' ' << relativePath_ << ' ' << location_ << ' ' << canonicalFilename_.substr(localTop_.size());
    } else if (location_ == Release) {
      // Guarantee a site independent value by stripping $RELEASETOP.
      if (releaseTop_.empty()) {
	throw edm::Exception(edm::errors::FileInPathError)
	  << "Environment Variable " 
	  << RELEASETOP
	  << " is not set.\n";
      }
      std::string::size_type pos = canonicalFilename_.find(releaseTop_);
      if (pos != 0) {
	throw edm::Exception(edm::errors::FileInPathError)
	  << "Path " 
	  << canonicalFilename_
	  << " is not in the base release area "
	  << releaseTop_
	  << "\n";
      }
      os << version << ' ' << relativePath_ << ' ' << location_ << ' ' << canonicalFilename_.substr(releaseTop_.size());
    } else if (location_ == Data) {
      // Guarantee a site independent value by stripping $DATATOP.
      if (dataTop_.empty()) {
	throw edm::Exception(edm::errors::FileInPathError)
	  << "Environment Variable " 
	  << DATATOP
	  << " is not set.\n";
      }
      std::string::size_type pos = canonicalFilename_.find(dataTop_);
      if (pos != 0) {
	throw edm::Exception(edm::errors::FileInPathError)
	  << "Path " 
	  << canonicalFilename_
	  << " is not in the data area "
	  << dataTop_
	  << "\n";
      }
      os << version << ' ' << relativePath_ << ' ' << location_ << ' ' << canonicalFilename_.substr(dataTop_.size());
    }
  }


  void
  FileInPath::read(std::istream& is)
  {
    std::string vsn;
    std::string relname;
    std::string canFilename;
#if 1
    // This #if needed for backward compatibility
    // for files written before CMSSW_1_5_0_pre3.
    is >> vsn;
    if (!is) return;
    bool oldFormat = (version != vsn);
    if (oldFormat) {
      relname = vsn;
      bool local;
      is >> local;
      location_ = (local ? Local : Release);
      is >> canFilename;
    } else {
      // Current format
      int loc;
      is >> relname >> loc;
      location_ = static_cast<FileInPath::LocationCode>(loc);
      if (location_ != Unknown) is >> canFilename;
    }
#else
    is >> vsn >> relname >> loc >> canFilename;
#endif
    if (!is) return;
    relativePath_ = relname;
    if (location_ == Local) {
      if (localTop_.empty()) {
	throw edm::Exception(edm::errors::FileInPathError)
	  << "Environment Variable " 
	  << LOCALTOP
	  << " is not set.\n";
      }
#if 1
      // This #if needed for backward compatibility
      // for files written before CMSSW_1_5_0_pre3.
      if (oldFormat) {
        canonicalFilename_ = canFilename;
      } else
#endif
      canonicalFilename_ = localTop_ + canFilename;
    } else if (location_ == Release) {
      if (releaseTop_.empty()) {
	throw edm::Exception(edm::errors::FileInPathError)
	  << "Environment Variable " 
	  << RELEASETOP
	  << " is not set.\n";
      }
#if 1
      // This #if needed for backward compatibility
      // for files written before CMSSW_1_5_0_pre3.
      if (oldFormat) {
         std::string::size_type pos = canFilename.find(BASE);
        if (pos == 0) {
          // Replace the placehoder with the path to the base release (site dependent).
          canonicalFilename_ = releaseTop_ + canFilename.substr(BASE.size());
        } else {
          // Needed for files written before CMSSW_1_2_0_pre2.
          canonicalFilename_ = canFilename;
        }
      } else
#endif
      canonicalFilename_ = releaseTop_ + canFilename;
    } else if (location_ == Data) {
      if (dataTop_.empty()) {
	throw edm::Exception(edm::errors::FileInPathError)
	  << "Environment Variable " 
	  << DATATOP
	  << " is not set.\n";
      }
      canonicalFilename_ = dataTop_ + canFilename;
    }
  }

  //------------------------------------------------------------


  void 
  FileInPath::getEnvironment() {
    if (!envstring(RELEASETOP, releaseTop_)) {
      releaseTop_.clear();
    }
    if (releaseTop_.empty()) {
      // RELEASETOP was not set.  This means that the environment is set
      // for the base release itself.  So LOCALTOP actually contains the 
      // location of the base release.
      if (!envstring(LOCALTOP, releaseTop_)) {
        releaseTop_.clear();
      }
    } else {
      if (!envstring(LOCALTOP, localTop_)) {
        localTop_.clear();
      }
    }
    if (!envstring(DATATOP, dataTop_)) {
      dataTop_.clear();
    }
    if (!envstring(PathVariableName, searchPath_)) {
      throw edm::Exception(edm::errors::FileInPathError)
	<< PathVariableName
	<< " must be defined\n";
    }
  }

  void 
  FileInPath::initialize_()
  {
    if (relativePath_.empty())
      throw edm::Exception(edm::errors::FileInPathError)
	<< "Relative path must not be empty\n";

    // Find the file, based on the value of path variable.
    typedef std::vector<std::string> stringvec_t;
    stringvec_t  pathElements = tokenize(searchPath_, ":");
    stringvec_t::const_iterator it =  pathElements.begin();
    stringvec_t::const_iterator end = pathElements.end();
    while (it != end) {
      // Set the boost::fs path to the current element of
      // CMSSW_SEARCH_PATH:
      bf::path pathPrefix(*it, bf::no_check);

      // Does the a file exist? locateFile throws is it finds
      // something goofy.
      if (locateFile(pathPrefix, relativePath_)) {
	// Convert relative path to canonical form, and save it.
	relativePath_ = bf::path(relativePath_, bf::no_check).normalize().string();
	  
	// Save the absolute path.
	canonicalFilename_ = bf::complete(relativePath_, 
					  pathPrefix).string();
	if (canonicalFilename_.empty())
	  throw edm::Exception(edm::errors::FileInPathError)
	    << "fullPath is empty"
	    << "\nrelativePath() is: " << relativePath_
	    << "\npath prefix is: " << pathPrefix.string()
	    << '\n';

	// From the current path element, find the branch path (basically the path minus the
	// last directory, e.g. /src or /share):
	for (bf::path br = pathPrefix.branch_path(); !br.normalize().string().empty(); br = br.branch_path()) {

	  if (!localTop_.empty()) {
	    // Create a path object for our local path LOCALTOP:
	    bf::path local_(localTop_, bf::no_check);
	    // If the branch path matches the local path, the file was found locally:
	    if (br == local_) {
	      location_ = Local;
	      return;
	    }
	  }

	  if (!releaseTop_.empty()) {
	    // Create a path object for our release path RELEASETOP:
	    bf::path release_(releaseTop_, bf::no_check);
	    // If the branch path matches the release path, the file was found in the release:
	    if (br == release_) {
	      location_ = Release;
	      return;
	    }
	  }

	  if (!dataTop_.empty()) {
	    // Create a path object for our data path DATATOP:
	    bf::path data_(dataTop_, bf::no_check);
	    // If the branch path matches the data path, the file was found in the data area:
	    if (br == data_) {
	      location_ = Data;
	      return;
	    }
	  }
	}
	    
	// This is really gross --- this organization of if/else
	// inside the while-loop should be changed so that
	// this break isn't needed.
      }
      // Keep trying
      ++it;
    }
    
    // If we got here, we ran out of path elements without finding
    // what we're looking found.
    throw edm::Exception(edm::errors::FileInPathError)
      << "edm::FileInPath unable to find file "
      << relativePath_
      << " anywhere in the search path."
      << "\nThe search path is defined by: "
      << PathVariableName
      << "\n${"
      << PathVariableName
      << "} is: "
      << getenv(PathVariableName.c_str())
      << "\nCurrent directory is: "
      << bf::initial_path().string()
      << "\n";    
  }
    
  
}




// ----------------------------------------------------------------------
// $Id: FileInPath.cc,v 1.7.2.1 2005/12/08 11:21:26 sashby Exp $
//
// ----------------------------------------------------------------------

// TODO: This file needs some clean-up, especially regarding the
// handling of environment variables. We can do better with
// translating them only once --- after we have settled down on how
// long the search path is allowed to be, and whether are only choice
// for the "official" directory is CMSSW_DATA_PATH.

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/tokenizer.hpp"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"

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
        
  // Return false if the environment variable 'name is not found, and
  // true if it is found. If it is found, put the translation of the
  // environment variable into 'result'.
  bool envstring(std::string const& name,
		 std::string& result)
  {
    const char* val = getenv(name.c_str());
    if (val == 0)
      {
	// Temporary hackery to allow use of old names during
	// transition period.
	//
	// TODO: Remove these two "if" tests, leaving only the return
	// false, before the CMSSW_0_3_0 release
	if ( name == "CMSSW_DATA_PATH" ) 
	  return envstring("CMSDATA", result);
	if ( name == "CMSSW_SEARCH_PATH" ) 
	  return envstring("CMS_SEARCH_PATH", result);
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

    if ( bf::is_directory(p) )
      throw edm::Exception(edm::errors::FileInPathError)
	<< "Path " 
	<< p.native_directory_string()
	<< " is a directory, not a file\n";

    if ( bf::symbolic_link_exists(p) )
      throw edm::Exception(edm::errors::FileInPathError)
	<< "Path " 
	<< p.native_file_string()
	<< " is a symbolic link, not a file\n";

    return true;    
  }

}

namespace edm
{
  typedef boost::char_separator<char>   separator_t;
  typedef boost::tokenizer<separator_t> tokenizer_t;

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
    relativePath_( r ?
		   r :
		   ((throw edm::Exception(edm::errors::FileInPathError)
		    << "Relative path may not be null\n"), r)),
    canonicalFilename_(),
    isLocal_(false)
  {
    initialize_();    
  }

  FileInPath::FileInPath( FileInPath const& other) :
    relativePath_(other.relativePath_),
    canonicalFilename_(other.canonicalFilename_),
    isLocal_(other.isLocal_)
  {  }

  FileInPath&
  FileInPath::operator= (FileInPath const& other)
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
    std::swap(isLocal_, other.isLocal_);
  }

  std::string
  FileInPath::relativePath() const
  {
    return relativePath_;
  }


  bool
  FileInPath::isLocal() const
  {
    return isLocal_;
  }

  std::string
  FileInPath::fullPath() const
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
    initialize_();
  }

  //------------------------------------------------------------


  void 
  FileInPath::initialize_()
  {
    if (relativePath_.empty())
      throw edm::Exception(edm::errors::FileInPathError)
	<< "Relative path may not be empty\n";

    // Find the file, based on the value of path variable.
    std::string searchPath;
    if (!envstring(PathVariableName, searchPath))
      throw edm::Exception(edm::errors::FileInPathError)
	<< PathVariableName
	<< " must be defined\n";

    // boost::tokenizer is overkill here, but if we switch to allowing
    // a longer path (not just one or two entries), then it is useful.

    separator_t  sep(":"); // separator for elements in path
    tokenizer_t  tokens(searchPath, sep);

    typedef std::vector<std::string> stringvec_t;
    stringvec_t  pathElements;
    std::copy(tokens.begin(), 
	      tokens.end(),
	      std::back_inserter<stringvec_t>(pathElements));

    stringvec_t::const_iterator it =  pathElements.begin();
    stringvec_t::const_iterator end = pathElements.end();
    while (it != end)
      {
	bf::path pathPrefix;

	// Set the boost::fs path to the current element of
	// CMSSW_SEARCH_PATH:
	pathPrefix = *it;

	// Does the a file exist? locateFile throws is it finds
	// something goofy.
	if ( locateFile(pathPrefix, relativePath_) )
	  {
	  // Convert relative path to canonical form, and save it.
	  relativePath_ = bf::path(relativePath_).normalize().string();
	  
	  // Save the absolute path.
	  canonicalFilename_ = bf::complete(relativePath_, 
					      pathPrefix ).string();
	    if (canonicalFilename_.empty() )
	      throw edm::Exception(edm::errors::FileInPathError)
		<< "fullPath is empty"
		<< "\nrelativePath() is: " << relativePath_
		<< "\npath prefix is: " << pathPrefix.string()
		<< '\n';

	    // From the current path element, find the branch path (basically the path minus the
	    // last directory, e.g. /src or /share):
	    bf::path br = pathPrefix.branch_path();	   	    

	    std::string localtop_;
	    isLocal_ = false;

	    // Check that LOCALTOP really has a value and store it:
	    if (!envstring(LOCALTOP, localtop_))
		throw edm::Exception(edm::errors::FileInPathError)
		    << LOCALTOP
		    << " must be defined - is runtime environment set correctly?\n";

	    // Create a path object for our local path LOCALTOP:
	    bf::path local_ = localtop_;
	    
	    // If the branch path matches the local path, the file was found locally:
	    if (br == local_)
	       {
		   isLocal_ = true;
	       }
	    
	    // We're done...indeed.
	    
	    // This is really gross --- this organization of if/else
	    // inside the while-loop should be changed so that
	    // this break isn't needed.
	    return;
	  }
	// Keep trying
	++it;
      }
    
    // If we got here, we ran out of path elements without finding
    // what we're looking found.
    throw edm::Exception(edm::errors::FileInPathError)
      << "Unable to find file "
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





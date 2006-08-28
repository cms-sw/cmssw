#ifndef ParameterSet_FileInPath_h
#define ParameterSet_FileInPath_h

/// $Id: FileInPath.h,v 1.6 2005/11/15 14:38:57 paterno Exp $
///

/// Find a non-event-data file, given a relative path.

/// FileInPath knows how to take a string, interpreted as a relative
/// path to a file, and to interpret using the "standard CMS
/// non-event-data file searching mechanism".
///
/// The mechanism using the environment variables:
///    CMSSW_SEARCH_PATH: may be set by the end-user
///    CMSSW_DATA_PATH:         should be set by a site administrator
///
///  CMSSW_SEARCH_PATH is a 'search path' limited to either 1 or 2
///  components. The legal values are:
///
///
///       "." or "LOCAL", which means to search for files under
///            the top level of the "local working area", which is
///            defined as ${SCRAMRT_LOCALRT}/src
///
///       "CMSSW_DATA_PATH", which means search the "official place",
///             defined by the value of the CMSSW_DATA_PATH environment
///             variable, for files.
///
///       ".:CMSSW_DATA_PATH" or "LOCAL:CMSSW_DATA_PATH",
///              which means look first in the current working
///              directory, then in the "official place", for files.
///

// Notes:
//
//  1. We do not deal well with paths that contain spaces; this is because
//     of the way the ParameterSet system's 'encode' and 'decode' functions
//     are implemented for FileInPath objects. This could be fixed, if it
//     is important to handle filenames or paths with embedded spaces.


// TODO: Find the correct package for this class to reside. It
// doesn't seem well-suited for ParameterSet.


#include <istream>
#include <ostream>
#include <string>


namespace edm
{
  class FileInPath
  {
  public:

    /// Default c'tor does no file-existence check; what file would it
    /// check for existence?
    FileInPath();

    /// We throw an exception is the referenced file is not found.
    explicit FileInPath(const std::string& r);
    explicit FileInPath(const char* r);

    FileInPath(FileInPath const& other);
    FileInPath& operator=( FileInPath const& other);
    void swap(FileInPath& other);

    /// Return a string containing the canonical form of the
    /// *relative* path. DO NOT USE THIS AS THE FILENAME for any file
    /// operations; use fullPath() for that purpose.
    std::string relativePath() const;

    /// Was the file found under the "local" area?
    bool isLocal() const;

    /// Return a string that can be used to open the referenced
    /// file. 
    ///
    /// Note that operations on this file may fail, including
    /// testing for existence. This is because the state of a
    /// filesystem is global; other threads, processes, etc., may have
    /// removed the file since we checked on its existence at the time
    /// of construction of the FileInPath object.
    std::string fullPath() const;

    /// Write contents to the given ostream.
    /// Writing errors are reflected in the state of the stream.
    void write(std::ostream& os) const;
    
    /// Read from the given istream, and set contents accordingly.
    /// Reading errors are reflected in the state of the stream.
    void read(std::istream& is);

  private:
    std::string    relativePath_;
    std::string    canonicalFilename_;
    bool           isLocal_;


    // Helper function for construction.
    void initialize_();
  };

  // Free swap function
  inline
  void
  swap(FileInPath& a, FileInPath& b) 
  {
    a.swap(b);
  }

  inline  std::ostream& 
  operator<< (std::ostream& os, const edm::FileInPath& fip)
  {
    fip.write(os);
    return os;
  }

  inline std::istream&
  operator>> (std::istream& is, FileInPath& fip)
  {
    fip.read(is);
    return is;
  }

  inline bool
  operator== (edm::FileInPath const& a,
	      edm::FileInPath const& b)
  {
    return a.isLocal() == b.isLocal() && a.relativePath() == b.relativePath();      
  }

}

#endif

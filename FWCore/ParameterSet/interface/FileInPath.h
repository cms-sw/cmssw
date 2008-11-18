#ifndef FWCore_ParameterSet_FileInPath_h
#define FWCore_ParameterSet_FileInPath_h

///

/// Find a non-event-data file, given a relative path.

/// FileInPath knows how to take a string, interpreted as a relative
/// path to a file, and to interpret using the "standard CMS
/// non-event-data file searching mechanism".
///
/// The mechanism using the environment variables:
///    CMSSW_SEARCH_PATH:       may be set by the end-user
///    CMSSW_RELEASE_BASE:      should be set by a site administrator
///    CMSSW_DATA_PATH:         should be set by a site administrator
///
///  CMSSW_SEARCH_PATH is a 'search path' limited to 1 to 3
///  components. The legal values are:
///
///
///       "." or "LOCAL", which means to search for files under
///            the top level of the "local working area", which is
///            defined as ${SCRAMRT_LOCALRT}/src
///
///       "CMSSW_RELEASE_BASE", which means search the "official place",
///             defined by the value of the CMSSW_RELEASE_BASE environment
///             variable, for files.
///
///       "CMSSW_DATA_PATH", which means search the "official place",
///             defined by the value of the CMSSW_DATA_PATH environment
///             variable, for files.
///
///       ".:CMSSW_RELEASE_BASE" or "LOCAL:CMSSW_RELEASE_BASE",
///              which means look first in the current working
///              directory, then in the "official place", for files.
///
///       ".:CMSSW_DATA_PATH" or "LOCAL:CMSSW_DATA_PATH",
///              which means look first in the current working
///              directory, then in the "official place", for files.
///
///       ".:CMSSW_RELEASE_BASE:CMSSW_DATA_PATH" or "LOCAL:CMSSW_RELEASE_BASE:CMSSW_DATA_PATH",
///              which means look first in the current working
///              directory, then in both "official places", for files.
///

// Notes:
//
//  1. We do not deal well with paths that contain spaces; this is because
//     of the way the ParameterSet system's 'encode' and 'decode' functions
//     are implemented for FileInPath objects. This could be fixed, if it
//     is important to handle filenames or paths with embedded spaces.
//
//  2. All environment variables are read only once, when the FileInPath object is constructed.
//     Therefore, any changes made to these variables externally during the lifetime of
//     a FileInPath object will have no effect.


// TODO: Find the correct package for this class to reside. It
// doesn't seem well-suited for ParameterSet.


#include <iosfwd>
#include <string>


namespace edm
{
  class FileInPath
  {
  public:

    enum LocationCode {
      Unknown = 0,
      Local = 1,
      Release = 2,
      Data = 3
    };

    /// Default c'tor does no file-existence check; what file would it
    /// check for existence?
    FileInPath();

    /// We throw an exception is the referenced file is not found.
    explicit FileInPath(const std::string& r);
    explicit FileInPath(const char* r);

    FileInPath(FileInPath const& other);
    FileInPath& operator=( FileInPath const& other);
    ~FileInPath();
    void swap(FileInPath& other);

    /// Return a string containing the canonical form of the
    /// *relative* path. DO NOT USE THIS AS THE FILENAME for any file
    /// operations; use fullPath() for that purpose.
    std::string relativePath() const;

    /// Where was the file found?
    LocationCode location() const;

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

    /// for boost::serialization
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & relativePath_;
      ar & canonicalFilename_;
      ar & location_;
      ar & localTop_;
      ar & releaseTop_;
      ar & dataTop_;
      ar & searchPath_;
    }



  private:
    std::string    relativePath_;
    std::string    canonicalFilename_;
    LocationCode   location_;
    std::string    localTop_;
    std::string    releaseTop_;
    std::string    dataTop_;
    std::string    searchPath_;

    // Helper function for construction.
    void getEnvironment();
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
    return a.location() == b.location() && a.relativePath() == b.relativePath();      
  }

}

#endif

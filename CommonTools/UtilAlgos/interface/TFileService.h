#ifndef UtilAlgos_TFileService_h
#define UtilAlgos_TFileService_h
/* \class TFileService
 *
 * \author Luca Lista, INFN
 *
 * Modified to run properly in the multithreaded Framework.
 * Any modules that use this service in multithreaded mode
 * must be type "One" modules that declared a shared resource
 * of type TFileService::kSharedResource.
 *
 */

#include "CommonTools/Utils/interface/TFileDirectory.h"

#include <string>

class TDirectory;
class TFile;

namespace edm {
  class ActivityRegistry;
  class GlobalContext;
  class ModuleCallingContext;
  class ModuleDescription;
  class ParameterSet;
  class StreamContext;
}

class TFileService {
public:
  /// constructor
  TFileService(const edm::ParameterSet &, edm::ActivityRegistry &);
  /// destructor
  ~TFileService();
  /// return opened TFile
  TFile & file() const { return * file_; }

  /// Hook for writing info into JR
  void afterBeginJob();

  TFileDirectory & tFileDirectory() { return tFileDirectory_; }

  // The next 6 functions do nothing more than forward function calls
  // to the TFileDirectory data member.

  // cd()s to requested directory and returns true (if it is not
  // able to cd, it throws exception).
  bool cd() const { return tFileDirectory_.cd(); }

  // returns a TDirectory pointer
  TDirectory *getBareDirectory (const std::string &subdir = "") const {
    return tFileDirectory_.getBareDirectory(subdir);
  }

  // reutrns a "T" pointer matched to objname
  template< typename T > T* getObject (const std::string &objname,
                                       const std::string &subdir = "") {
    return tFileDirectory_.getObject<T>(objname, subdir);
  }

  /// make new ROOT object
  template<typename T, typename ... Args>
  T* make(const Args& ... args) const {
    return tFileDirectory_.make<T>(args ...);
  }

  /// create a new subdirectory
  TFileDirectory mkdir( const std::string & dir, const std::string & descr = "" ) {
    return tFileDirectory_.mkdir(dir, descr);
  }

  /// return the full path of the stored histograms
  std::string fullPath() const { return tFileDirectory_.fullPath(); }

  static const std::string kSharedResource;

private:
  static thread_local TFileDirectory tFileDirectory_;
  /// pointer to opened TFile
  TFile * file_;
  std::string fileName_;
  bool fileNameRecorded_;
  bool closeFileFast_;

  // set current directory according to module name and prepair to create directory
  void setDirectoryName( const edm::ModuleDescription & desc );
  void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void preModuleGlobal(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobal(edm::GlobalContext const&, edm::ModuleCallingContext const&);
};

namespace edm {
  namespace service {
    // This function is needed so that there will be only one instance
    // of this service per process when "subprocesses" are being used.
    inline
    bool isProcessWideService(TFileService const*) {
      return true;
    }
  }
}
#endif

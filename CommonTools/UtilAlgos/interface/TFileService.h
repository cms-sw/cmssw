#ifndef UtilAlgos_TFileService_h
#define UtilAlgos_TFileService_h
/* \class TFileService
 *
 * \author Luca Lista, INFN
 *
 */
#include "CommonTools/Utils/interface/TFileDirectory.h"

namespace edm {
  class ActivityRegistry;
  class ParameterSet;
  class ModuleDescription;
}

class TFileService : public TFileDirectory {
public:
  /// constructor
  TFileService(const edm::ParameterSet &, edm::ActivityRegistry &);
  /// destructor
  ~TFileService();
  /// return opened TFile
  TFile & file() const { return * file_; }

  /// Hook for writing info into JR
  void afterBeginJob();

private:
  /// pointer to opened TFile
  TFile * file_;
  std::string fileName_;
  bool fileNameRecorded_;
  bool closeFileFast_;
  // set current directory according to module name and prepair to create directory
  void setDirectoryName( const edm::ModuleDescription & desc );
};

namespace edm {
   namespace service {
    // This function is needed so that there will be only on instance
    // of this service per process when "subprocesses" are being used.
    inline
    bool isProcessWideService(TFileService const*) {
      return true;
    }
  }
}

#endif


#ifndef FWLite_TFileService_h
#define FWLite_TFileService_h
/* \class fwlite::TFileService
 *
 * \author Benedikt Hegner, CERN
 *
 */
#include "CommonTools/Utils/interface/TFileDirectory.h"

namespace fwlite {

class TFileService : public TFileDirectory {
 public:
  /// constructor
  TFileService(const std::string& fileName);

  /// constructor with external TFile
  TFileService(TFile * aFile);

  /// destructor
  ~TFileService() override;

  /// return opened TFile
  TFile & file() const { return * file_; }

 private:
  /// pointer to opened TFile
  TFile * file_;
  std::string fileName_;

};

}
#endif

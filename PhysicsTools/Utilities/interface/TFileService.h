#ifndef Utilities_TFileService_h
#define Utilities_TFileService_h
/* \class TFileService
 *
 * \author Luca Lista, INFN
 *
 */
#include "TFile.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

class TFileService {
public:
  /// constructor
  TFileService( const edm::ParameterSet &, edm::ActivityRegistry & );
  /// destructor
  ~TFileService();
  /// return opened TFile
  TFile & file() const { return * file_; } 

private:
  /// pointer to opened TFile
  TFile * file_;
};

#endif

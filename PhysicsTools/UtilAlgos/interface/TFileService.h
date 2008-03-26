#ifndef UtilAlgos_TFileService_h
#define UtilAlgos_TFileService_h
/* \class TFileService
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"

namespace edm {
  class ActivityRegistry;
  class ParameterSet;
  class ModuleDescription;
}

class TFileService : public TFileDirectory {
public:
  /// constructor
  TFileService( const edm::ParameterSet &, edm::ActivityRegistry & );
  /// destructor
  ~TFileService();
  /// return opened TFile
  TFile & file() const { return * file_; } 
  void cd( const std::string & ) const;

private:
  /// pointer to opened TFile
  TFile * file_;
  // set current directory according to module name and prepair to create directory
  void setDirectoryName( const edm::ModuleDescription & desc );
};

#endif

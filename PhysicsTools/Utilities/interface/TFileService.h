#ifndef Utilities_TFileService_h
#define Utilities_TFileService_h
/* \class TFileService
 *
 * \author Luca Lista, INFN
 *
 */
#include <string>

class TFile;

namespace edm {
  class ActivityRegistry;
  class ParameterSet;
  class ModuleDescription;
}


class TFileService {
public:
  /// constructor
  TFileService( const edm::ParameterSet &, edm::ActivityRegistry & );
  /// destructor
  ~TFileService();
  /// return opened TFile
  TFile & file() const { return * file_; } 
  /// create current directory
  void mkdir() const;
  /// change to current directory
  void cd() const;
    
private:
  /// pointer to opened TFile
  TFile * file_;
  // set current directory according to module name and prepair to create directory
  void preModuleConstructor( const edm::ModuleDescription & desc );
  // set current directory according to module name
  void preModule( const edm::ModuleDescription & desc );
  // current module label
  std::string currentModuleLabel_, currentModulenName_;
};

#endif

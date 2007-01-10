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
  /// make new ROOT object
  template<typename T>
  T * make() const {
    cd(); return new T();
  }
  template<typename T, typename Arg1>
  T * make( const Arg1 & a1 ) const {
    cd(); return new T( a1 );
  }
  template<typename T, typename Arg1, typename Arg2>
  T * make( const Arg1 & a1, const Arg2 & a2 ) const {
    cd(); return new T( a1, a2 );
  }
  template<typename T, typename Arg1, typename Arg2, typename Arg3>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3 ) const {
    cd(); return new T( a1, a2, a3 );
  }
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4 ) const {
    cd(); return new T( a1, a2, a3, a4 );
  }
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5 ) const {
    cd(); return new T( a1, a2, a3, a4, a5 );
  }
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6 ) const {
    cd(); return new T( a1, a2, a3, a4, a5, a6 );
  }
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7 ) const {
    cd(); return new T( a1, a2, a3, a4, a5, a6, a7 );
  }
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8 ) const {
    cd(); return new T( a1, a2, a3, a4, a5, a6, a7, a8 );
  }
private:
  /// pointer to opened TFile
  TFile * file_;
  // set current directory according to module name and prepair to create directory
  void setDirectoryName( const edm::ModuleDescription & desc );
  // current module label
  std::string currentModuleLabel_, currentModulenName_;
  /// change (and possibly create) to current directory
  void cd() const;
};

#endif

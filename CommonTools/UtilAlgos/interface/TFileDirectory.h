#ifndef UtilAlgos_TFileDirectory_h
#define UtilAlgos_TFileDirectory_h
/* \class TFileDirectory
 *
 * \author Luca Lista, INFN
 *
 */
#include <string>
#include "CommonTools/UtilAlgos/interface/TH1AddDirectorySentry.h"

class TFile;
class TDirectory; 

class TFileDirectory {
public:
  /// descructor
  virtual ~TFileDirectory() { }
  /// make new ROOT object
  template<typename T>
  T * make() const {
    T* t = new T();
    t->SetDirectory(cd()); 
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1>
  T * make( const Arg1 & a1 ) const {
    T * t = new T( a1 );
    t->SetDirectory(cd());
    return t; 
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2>
  T * make( const Arg1 & a1, const Arg2 & a2 ) const {
    T * t =  new T( a1, a2 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3 ) const {
    T * t =  new T( a1, a2, a3 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4 ) const {
    T * t =  new T( a1, a2, a3, a4 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5 ) const {
    T * t =  new T( a1, a2, a3, a4, a5 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6 ) const {
    T * t =  new T( a1, a2, a3, a4, a5, a6 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7 ) const {
    T * t =  new T( a1, a2, a3, a4, a5, a6, a7 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8 ) const {
    T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
	   typename Arg9>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
	    const Arg9 & a9 ) const {
    T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8, a9 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
	   typename Arg9, typename Arg10>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
	    const Arg9 & a9, const Arg10 & a10 ) const {
    T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
	   typename Arg9, typename Arg10, typename Arg11>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
	    const Arg9 & a9, const Arg10 & a10, const Arg11 & a11 ) const {
    T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 );
    t->SetDirectory(cd());
    return t;
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
	   typename Arg9, typename Arg10, typename Arg11, typename Arg12>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
	    const Arg9 & a9, const Arg10 & a10, const Arg11 & a11, const Arg12 & a12 ) const {
    T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 );
    t->SetDirectory(cd());
    return t;
  }
  /// create a new subdirectory
  TFileDirectory mkdir( const std::string & dir, const std::string & descr = "" );

private:
  TFileDirectory( const std::string & dir, const std::string & descr,
		  TFile * file, const std::string & path ) : 
     file_( file ), dir_( dir ), descr_( descr ), path_( path ) {
  }
  friend class TFileService;
  TFile * file_;
  std::string dir_, descr_, path_;
  std::string fullPath() const;
  TDirectory * cd() const;
};

#endif

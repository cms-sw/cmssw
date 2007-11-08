#ifndef UtilAlgos_TFileDirectory_h
#define UtilAlgos_TFileDirectory_h
/* \class TFileDirectory
 *
 * \author Luca Lista, INFN
 *
 */
#include <string>
#include "PhysicsTools/UtilAlgos/interface/TH1AddDirectorySentry.h"

class TFile;

class TFileDirectory {
public:
  /// descructor
  virtual ~TFileDirectory() { }
  /// make new ROOT object
  template<typename T>
  T * make() const {
    TH1AddDirectorySentry sentry;
    cd(); return new T();
  }
  /// make new ROOT object
  template<typename T, typename Arg1>
  T * make( const Arg1 & a1 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2>
  T * make( const Arg1 & a1, const Arg2 & a2 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3, a4 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3, a4, a5 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3, a4, a5, a6 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3, a4, a5, a6, a7 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3, a4, a5, a6, a7, a8 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
	   typename Arg9>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
	    const Arg9 & a9 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3, a4, a5, a6, a7, a8, a9 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
	   typename Arg9, typename Arg10>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
	    const Arg9 & a9, const Arg10 & a10 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
	   typename Arg9, typename Arg10, typename Arg11>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
	    const Arg9 & a9, const Arg10 & a10, const Arg11 & a11 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 );
  }
  /// make new ROOT object
  template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
	   typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
	   typename Arg9, typename Arg10, typename Arg11, typename Arg12>
  T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
	    const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
	    const Arg9 & a9, const Arg10 & a10, const Arg11 & a11, const Arg12 & a12 ) const {
    TH1AddDirectorySentry sentry;
    cd(); return new T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 );
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
  void cd() const;
};

#endif

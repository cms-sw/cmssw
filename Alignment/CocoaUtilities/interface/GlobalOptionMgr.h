//   COCOA class header file
//Id:  GlobalOptionMgr.h
//CAT: Model
//
//   Class to manage the sets of fitted entries (one set per each measurement data set)
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _GlobalOptionMgr_HH
#define _GlobalOptionMgr_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <map>

class ALIFileIn;

class GlobalOptionMgr
{
private:
  GlobalOptionMgr(){ };

public:
  //---------- Constructors / Destructor
  ~GlobalOptionMgr(){ };
  static GlobalOptionMgr* getInstance();

 public:
  /// Set the list of default global options 
  void setDefaultGlobalOptions();

  std::map< ALIstring, ALIdouble, std::less<ALIstring> >& GlobalOptions() {
    return theGlobalOptions;
  }

  ALIdouble getGlobalOption( const ALIstring& sstr );

  ///----- Search a string in theGlobalOptions and return 1 if found
  int getGlobalOptionValue( const ALIstring& sstr, ALIdouble& val );
  void setGlobalOption( const ALIstring go, const ALIdouble val, ALIFileIn& filein );
  bool setGlobalOption( const ALIstring go, const ALIdouble val, bool bExit = 1 );
 
private:
  static GlobalOptionMgr* theInstance;
  /// global options
  std::map< ALIstring, ALIdouble, std::less<ALIstring> > theGlobalOptions;

};

#endif


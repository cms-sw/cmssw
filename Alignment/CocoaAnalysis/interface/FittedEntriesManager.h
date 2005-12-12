//   COCOA class header file
//Id:  FittedEntriesManager.h
//CAT: Model
//
//   Class to manage the sets of fitted entries (one set per each measurement data set)
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _FittedEntriesManager_HH
#define _FittedEntriesManager_HH

#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "OpticalAlignment/CocoaAnalysis/interface/FittedEntriesSet.h"


class FittedEntriesManager
{

public:
  //---------- Constructors / Destructor
  FittedEntriesManager(){ };
  ~FittedEntriesManager(){ };
  static FittedEntriesManager* getInstance();  
  void AddFittedEntriesSet( FittedEntriesSet* fents);  
  void MakeHistos();

 public:
  std::vector< FittedEntriesSet* > getFittedEntriesSets() const {
    return  theFittedEntriesSets; }

private:
  ALIstring createFileName( const ALIstring& optoName, const ALIstring& entryName);
  void dumpEntriesSubstraction( std::ofstream& fout, FittedEntriesSet& fes, ALIint order1, ALIint order2 );

private:
  static FittedEntriesManager* instance;
  std::vector< FittedEntriesSet* > theFittedEntriesSets;

  void GetDifferentBetweenLasers();

};

#endif


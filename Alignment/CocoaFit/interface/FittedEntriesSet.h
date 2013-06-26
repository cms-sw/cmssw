//   COCOA class header file
//Id:  FittedEntriesSet.h
//CAT: Model
//
//   Class to store set of fitted entries with date 
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _FittedEntriesSet_HH
#define _FittedEntriesSet_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "time.h"
#include <vector>
#include "Alignment/CocoaFit/interface/FittedEntry.h"
//#ifdef MAT_MESCHACH
#include "Alignment/CocoaFit/interface/MatrixMeschach.h"
//typedef MatrixMeschach ALIMatrix;
//#endif

class FittedEntriesSet
{

public:
  //---------- Constructors / Destructor
  FittedEntriesSet( MatrixMeschach* AtWAMatrix );
  FittedEntriesSet( std::vector<ALIstring> wl );
  //---- Average a list of FittedEntriesSet's
  FittedEntriesSet( std::vector<FittedEntriesSet*> vSets ); 
  ~FittedEntriesSet(){ };
  void Fill();
  void FillEntries();
  void FillCorrelations();
  void CreateCorrelationMatrix( const ALIuint nent );
  void FillEntriesFromFile( std::vector<ALIstring> wl);
  void FillEntriesAveragingSets( std::vector<FittedEntriesSet*> vSets );

  void SetOptOEntries();

 public:
  std::vector< FittedEntry* >& FittedEntries(){
    return theFittedEntries;
  }

//GET AND SET METHODS
  ALIstring& getDate() {
    return theDate;
  }
  ALIstring& getTime() {
    return theTime;
  }

public:

  std::vector< FittedEntry* > theFittedEntries;
 private:
//t  struct tm theTime;
  ALIstring theDate;
  ALIstring theTime;
  std::vector< std::vector<ALIdouble> > theCorrelationMatrix;
  ALIint theMinEntryQuality;
  MatrixMeschach* theEntriesErrorMatrix;
};

#endif


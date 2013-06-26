#ifndef MERunManager_hh
#define MERunManager_hh

#include "MusEcal.hh"
class MERun;
class MEChannel;
class MEVarVector;

class MERunManager 
{
public:

  // a run manager for a given monitoring region, run type, color
  MERunManager( unsigned int lmr, 
		unsigned int type, // e.g. ME::iLaser, iTestPulse, iPedestal 
		unsigned int color=ME::iSizeC ); // when type=iLaser

  virtual ~MERunManager();

  void updateRunList();

  // iterators
  MusEcal::RunIterator it();
  MusEcal::RunIterator it(   ME::Time );
  MusEcal::RunIterator from( ME::Time );
  MusEcal::RunIterator to(   ME::Time );
  MusEcal::RunIterator begin();
  MusEcal::RunIterator end();
  MusEcal::RunIterator first();
  MusEcal::RunIterator last();
  MusEcal::RunIterator cur();

  // keys
  ME::Time beginKey()     const;
  ME::Time endKey()       const;
  ME::Time firstKey()     const { return _first; }
  ME::Time lastKey()      const { return _last; }
  ME::Time normFirstKey() const { return _normFirst; }
  ME::Time normLastKey()  const { return _normLast; }
  ME::Time curKey()       const { return _current; }
  ME::Time closestKey( ME::Time key );
  ME::Time closestKeyInFuture( ME::Time key );

  // runs
  unsigned size() const { return _runs.size(); }
  MERun* beginRun();
  MERun*   endRun();
  MERun* firstRun();
  MERun*  lastRun();
  MERun*   curRun();
  MERun*  run( ME::Time );

  bool setCurrentRun( ME::Time key );
  void setNoCurrent();

  bool setPlotRange(  ME::Time first, ME::Time last, bool verbose=true );
  bool setNormRange(  ME::Time first, ME::Time last, bool verbose=true );

  // set the current run to bad
  void setBadRun();

  // set the range to bad
  void setBadRange( ME::Time firstBad, ME::Time lastBad, bool verbose=true  );

  // refresh bad runs
  void refreshBadRuns();

  // is a run good
  bool isGood( ME::Time );

  // tree of channels
  MEChannel* tree();
  
  void print();

  int LMRegion() const { return _lmr; }

  // fill the maps (should be private)
  void fillMaps();

  // set good/bad flags
  void setFlags();
  void setLaserFlags();
  void setTestPulseFlags();

  // get the vectors
  MEVarVector* apdVector( MEChannel* );
  MEVarVector* pnVector( MEChannel*, int ipn );
  MEVarVector* mtqVector( MEChannel* );

  // recursive flag setting 
  void setFlag( MEChannel* leaf, ME::Time time, int var, bool flag ); 

  void refresh();

private :
  
  int   _lmr;
  int   _reg;
  int    _sm;
  int   _dcc;
  int  _side;
  int  _type;
  int _color;

  // absolute paths to the LM data and to the primitives
  TString _lmdataPath;
  TString _primPath;

  MusEcal::RunMap _runs;

  ME::Time _first;
  ME::Time _last;
  ME::Time _current;
  ME::Time _normFirst;
  ME::Time _normLast;
  
  // map of bad runs
  MusEcal::BoolTimeMap _badRuns;

  // access to the data for each channel
  std::map< MEChannel*, MEVarVector* >  _apdMap;
  std::map< MEChannel*, MEVarVector* >  _pnMap[2];
  std::map< MEChannel*, MEVarVector* >  _mtqMap;

  MEChannel* _tree;  // the tree of channels

  ClassDef( MERunManager, 0 ) // MERunManager  -- manages laser monitoring runs
};

#endif

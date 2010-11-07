#ifndef METimeInterval_hh
#define METimeInterval_hh

#include <list>
#include <TString.h>

#include "../../interface/ME.h"

class METimeInterval 
{
public:

  static int instances;

  // public constructor
  METimeInterval( ME::Time firstTime, ME::Time lastTime );

  // public splitting
  void split( const std::list<ME::Time>& times );

  // recursive destructor
  virtual ~METimeInterval();
  
  // find an interval at a given level
  METimeInterval* get( ME::Time time, unsigned level=1000 );

  // first and last intervals at required level connected to this 
  METimeInterval* first( unsigned level );
  METimeInterval* last( unsigned level );

  // accessors
  unsigned level() { return _level; }
  ME::Time firstTime() const { return _firstTime; }
  ME::Time lastTime()  const { return _lastTime; }
  bool isGood() const { return _good; }

  // interval navigation at same level
  METimeInterval* above() { return _above; }
  METimeInterval* next();
  METimeInterval* previous();
  METimeInterval* firstIn();
  METimeInterval* lastIn();

  // recursive printing
  void print( unsigned level );

  // one-line printing
  void oneLine();

  // convenient printing [time0;time1[
  TString inBrackets();

private :

  // private constructor
  METimeInterval( ME::Time firstTime, ME::Time lastTime, METimeInterval* previous, METimeInterval* above, bool OK=true ); 

  // private splitting
  METimeInterval* cut( ME::Time time );

  // time interval
  ME::Time _firstTime;
  ME::Time _lastTime;

  // navigation pointers
  METimeInterval* _previous;
  METimeInterval* _next;
  METimeInterval* _above;
  METimeInterval* _below;

  // level of the interval, 0 being the top interval
  unsigned _level;

  bool _good;

  ClassDef(METimeInterval,0) // MusEcal: a time interval
};

#endif

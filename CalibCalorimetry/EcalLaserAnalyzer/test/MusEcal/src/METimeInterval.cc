#define METimeInterval_cxx
#include <assert.h>
#include <fstream>
#include <list>
#include <iostream>
#include <algorithm>
using namespace std;
#include <TString.h>

#include "METimeInterval.hh"

ClassImp( METimeInterval )

int METimeInterval::instances=0;
 
// public constructor
METimeInterval::METimeInterval( ME::Time firstTime, ME::Time lastTime )
  : _firstTime( firstTime ), _lastTime (lastTime ),
    _previous(0), _next(0), _above(0), _below(0), _level(0), _good(true)
{
  assert( _lastTime>_firstTime );
  instances++;
}

// private constructor
METimeInterval::METimeInterval( ME::Time firstTime, ME::Time lastTime, METimeInterval* previous, METimeInterval* above, bool good )
  : _firstTime(firstTime), _lastTime(lastTime),
    _previous(previous), _next(0), _above(above), _below(0), _good(good)
{
  instances++;
  if( _previous==0 ) 
    {
      assert( _above!=0 );
      assert( _firstTime == _above->_firstTime );
      assert( _lastTime == _above->_lastTime );
      _level = _above->_level + 1;
    }
  else
    {
      _level  = _previous->_level;
    }
  assert( _lastTime>_firstTime );
}

METimeInterval::~METimeInterval()
{
  if( _next )  delete _next;  _next=0;
  if( _below ) delete _below; _below=0;
  if( _previous ) _previous->_next=0;
  else if( _above ) _above->_below=0;
  instances--;
}

void
METimeInterval::split( const list<ME::Time>& times )
{  
  if( _below!=0 ) _below->split( times );
  else
    {
      _below = new METimeInterval( _firstTime, _lastTime, 0, this );
      METimeInterval* current = _below;
      list<ME::Time> theList( times );
      theList.unique();
      theList.sort();
      list<ME::Time>::iterator it;
      for( it=theList.begin(); it!=theList.end(); it++ )
	{
	  ME::Time time = *it;
	  if( time<=_firstTime ) continue;
	  if( time>=_lastTime  ) break;
	  current = current->cut( time );
	}  
    }
  if( _next ) _next->split( times );
}

METimeInterval*
METimeInterval::cut( ME::Time time )
{
  if( time<=_firstTime || time>=_lastTime ) return this;
  _next = new METimeInterval( time, _lastTime, this, _above );
  _lastTime = time;  
  return _next;
}

void 
METimeInterval::print( unsigned level )
{
  if( _level==level )
    {
      oneLine();
    }
  if( _below ) _below->print( level );
  if( _next  ) _next->print( level );
}

void 
METimeInterval::oneLine()
{
  cout << "firstTime/lastTime/level ";
  cout << _firstTime << "/" << _lastTime << "/" << _level << endl;
}

TString
METimeInterval::inBrackets()
{
  TString str;
  str += "[";
  str += _firstTime;
  str += ";";
  str += _lastTime;
  str += "[";
  return str;
}

METimeInterval*
METimeInterval::get( ME::Time time, unsigned level )
{
   if( _level>level )
     {
       assert( _above!=0 );
       return _above->get( time, level );
     }
   if( time<_firstTime )
    {
      if( _previous ) return _previous->get( time, level );
      return 0;
    }
  if( time>=_lastTime ) 
    {
      if( _next ) return _next->get( time, level );
      if( time==_lastTime )
	{
	  //	  if( _below ) return _below->get( time, level );
	  return 0;  // last run is excluded !
	}
      return 0;
    }
  if( _level==level ) return this;
  if( _below ) return _below->get( time, level );
  return this; // this is as far as it gets
}

METimeInterval*
METimeInterval::first( unsigned level )
{
  if( _level<level ) 
    {
      if( _below ) return _below->first( level ); 
      return 0;
    }
  if( _level>level ) 
    {
      if( _above ) return _above->first( level ); 
      return 0;
    }
  return this->firstIn();
}

METimeInterval*
METimeInterval::last( unsigned level )
{
  if( _level<level ) 
    {
      if( _below->lastIn() ) return _below->lastIn()->last( level ); 
      return 0;
    }
  if( _level>level ) 
    {
      if( _above->lastIn() ) return _above->lastIn()->last( level ); 
      return 0;
    }
  return this->lastIn();
}

METimeInterval*
METimeInterval::firstIn()
{
  if( _previous ) return _previous->firstIn(); 
  return this;
}
METimeInterval*
METimeInterval::lastIn()
{
  if( _next ) return _next->lastIn(); 
  return this;
}

METimeInterval*
METimeInterval::next()
{
  if( _next ) return _next; 
  if( _above==0 ) return 0;
  METimeInterval* _above_next = _above->next();
  if( _above_next==0 ) return 0;
  METimeInterval* _above_next_below = _above_next->_below;
  if( _above_next_below==0 ) return 0;
  return _above_next_below->firstIn();
}

METimeInterval*
METimeInterval::previous()
{
  if( _previous ) return _previous; 
  if( _above!=0 && _above->_previous!=0 && _above->_previous->_below!=0 ) 
    return _above->_previous->_below->lastIn();
  return 0;
}

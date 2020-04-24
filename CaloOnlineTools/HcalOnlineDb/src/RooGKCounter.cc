#include "CaloOnlineTools/HcalOnlineDb/interface/RooGKCounter.h"

#include <iostream>
#include <cmath>
#include <cstdio>
using std::cout;
using std::endl;
using std::ostream;

//ClassImp(RooGKCounter)

RooGKCounter::RooGKCounter(){
  init();
} 

RooGKCounter::RooGKCounter( const char *message ){
  init();
  _message = message;
  if ( _message . length() == 0 ) printCount = false;
} 

RooGKCounter::RooGKCounter( unsigned long int theFirst, unsigned long int theDivider ){
  init();
  _count = theFirst;
  _firstCount = theFirst;
  _divider = theDivider;
  printCount = true;
} 

void RooGKCounter::setCounter( unsigned long int theCount ){
  _count = theCount;
}

void RooGKCounter::setDivider( unsigned int theDivider ){
  _divider = theDivider;
}

void RooGKCounter::setPrintCount( bool _printCount ){
  printCount = _printCount;
}

void RooGKCounter::setNewLine( bool newLine ){
  _newLine = newLine;
}

void RooGKCounter::setMessage( const char *message ){
  _message = message;
}

void RooGKCounter::init( void ){
  _count = 0;
  _firstCount = 0;
  _divider = 1;
  printCount = false;
  firstCountEntry = true;
  _message = "processing entry #";
  _newLine = true;

  initTime = time( nullptr );
  firstTickTime = 1;
  lastTickTime = 1;
  lastPrintTime = 1;

}

void RooGKCounter::count( void ){

  _count++;

  double _number;
  double _freq;
  double _limit = 1./(double)_divider;

  _number = (double)_count;
  _freq = (double)_divider;

  if (firstCountEntry){
    if ( printCount ) std::cout << "Counter is on:" << std::endl;
    firstCountEntry = false;
    firstTickTime = time( nullptr );
  }

  if ( printCount ){
    if ( fmod( _number, _freq ) < _limit ){
      double averageTimeSinceFirstTick = 0.0;
      if ( _count > _firstCount )
	{
	  averageTimeSinceFirstTick = ( time( nullptr ) - firstTickTime ) / (double)( _count - _firstCount );
	}
      if ( !_newLine )
	{
	  std::cout << char(13) << _message . c_str() << _count;
	  if ( _count > _firstCount ) std::cout << ", average time per count, sec: " << averageTimeSinceFirstTick;
	  fflush(stdout);
	}
      else
	{
	  std::cout << _message . c_str() << _count;
	  if ( _count > _firstCount ) std::cout << ", average time per count, sec: " << averageTimeSinceFirstTick;
	  std::cout << std::endl;
	}
      lastPrintTime = time( nullptr );
    }
  }
  
  lastTickTime = time( nullptr );
}

unsigned long int RooGKCounter::getCount( void ){
  return _count;
}

void RooGKCounter::increment( long int _incr ){
  _count += _incr;
}

RooGKCounter::~RooGKCounter(){
}

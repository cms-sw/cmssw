/* -*- mode: c++ -*- */

#ifndef Roo_GKCounter
#define Roo_GKCounter

#include <iostream>
#include <string>
#include <time.h>

class RooGKCounter {
  
 public:
  RooGKCounter();
  RooGKCounter( const char *message );
  RooGKCounter( unsigned long int theFirst, unsigned long int theDivider );
  ~RooGKCounter();

  void setCounter( unsigned long int );
  void setDivider( unsigned int );
  void setPrintCount( bool _printCount );
  void setNewLine( bool newLine );
  void setMessage( const char* );
  void count( void );
  unsigned long int getCount( void );
  void increment( long int _incr );

 private:
  unsigned long int _count;
  unsigned long int _firstCount;
  unsigned int _divider;
  bool printCount;
  bool firstCountEntry;
  std::string _message;
  bool _newLine;

  time_t initTime, firstTickTime, lastPrintTime, lastTickTime;

  void init( void );

  //ClassDef(RooGKCounter,1)
};

#endif

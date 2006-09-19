/** \file
 * 
 *  $Date: 2006/09/06 18:50:07 $
 *
 * \author FRC
 */


#include <DataFormats/DTDigi/interface/DTLocalTrigger.h>

#include <iostream>

using namespace std;


DTLocalTrigger::DTLocalTrigger (int bx, int qual) : 

  theBX (bx),
  theQuality(qual)
{}


DTLocalTrigger::DTLocalTrigger () : 

  theBX (0),
  theQuality(0)
{}


// Comparison
bool DTLocalTrigger::operator == (const DTLocalTrigger& trig) const {
  if ( theBX != trig.bx() ||
       theQuality != trig.quality() ) return false;
  return true;
}

// Getters


uint16_t DTLocalTrigger::bx() const { return theBX; }

uint16_t DTLocalTrigger::quality() const { return theQuality; }

// Setters ??

// Debug

void
DTLocalTrigger::print() const {
  cout << " trigger at BX "<<bx() 
       << " Quality "<<quality() << endl;
}


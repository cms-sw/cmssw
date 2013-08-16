/** \file
 * 
 *  $Date: 2008/01/22 18:45:04 $
 *
 * \author FRC
 */


#include <DataFormats/DTDigi/interface/DTLocalTrigger.h>


using namespace std;


DTLocalTrigger::DTLocalTrigger (int eventBx, int bx, int data) : 
  theEventBX(eventBx),
  theBX (bx),
  theData(data)
{}


DTLocalTrigger::DTLocalTrigger () : 
  theEventBX(0),
  theBX (0),
  theData(0)
{}


// Comparison
bool DTLocalTrigger::operator == (const DTLocalTrigger& trig) const {
  if ( theBX != trig.bx() ||
       this->quality() != trig.quality() ) return false;
  return true;
}

// Getters


uint16_t DTLocalTrigger::bx() const { return theBX; }

uint16_t DTLocalTrigger::quality() const {
  return ( (theData & 0xE) >> 1 );
}
uint16_t DTLocalTrigger::trTheta() const {
  return ( (theData & 0x30) >> 4 );
}

bool DTLocalTrigger::secondTrack() const {
  return ( theData & 0x1 );
}
bool DTLocalTrigger::trOut() const {
  return ( (theData & 0x40) >> 6 );
}

// Setters ??

// Debug

void
DTLocalTrigger::print() const {
  cout << " trigger at BX "<<bx()<<": "<<theData;   
  if (secondTrack()) 
    cout << " IT IS A SECOND TRACK !! ";
  cout << " Quality "<<quality();
  if (trTheta() == 1) 
    cout << " with a low Theta trigger ";
  if (trTheta() == 3) 
    cout << " with a high Theta trigger ";
  if (trOut()) 
    cout << " Trigger Out set ";
  cout << endl;
}

uint16_t DTLocalTrigger::eventBx() const {
  return theEventBX;
}

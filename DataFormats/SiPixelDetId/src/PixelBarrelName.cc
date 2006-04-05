#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

#include <sstream>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

using namespace std;

PixelBarrelName::PixelBarrelName(const DetId & id) 
  : PixelModuleName(true)
{
 
//  uint32_t rawId = id.rawId(); 
  PXBDetId cmssw_numbering(id);

  theLayer = cmssw_numbering.layer();
  theModule = cmssw_numbering.module() -4; if (theModule<=0) theModule--;
  theLadder = cmssw_numbering.ladder();
  if (theLayer == 1) {
    if (theLadder <= 5) theLadder -=6;
    else if (theLadder >= 6 && theLadder <= 15 ) theLadder -= 5;
    else if (theLadder >= 16) theLadder -= 26;
  } 
  else if (theLayer == 2) {
    if (theLadder <= 8) theLadder -= 9;
    else if (theLadder >= 9 && theLadder <= 24) theLadder -= 8;
    else if (theLadder >= 25) theLadder -= 41; 
  } 
  else if (theLayer == 3) {
    if (theLadder <= 11) theLadder -= 12;
    else if (theLadder >= 12 && theLadder <= 33) theLadder -= 11;
    else if (theLadder >= 34) theLadder -= 56;
  } else {
//    cout << " PROBLEM, no such layer " << endl;
  } 
}

bool PixelBarrelName::isFullModule() const
{
  bool result = true;
  if (abs(theLadder) == 1) result = false;
  if (theLayer == 1 && abs(theLadder) == 10) result = false;
  if (theLayer == 2 && abs(theLadder) == 16) result = false;
  if (theLayer == 3 && abs(theLadder) == 22) result = false;
  return result;
}
  
string PixelBarrelName::name() const 
{
   std::ostringstream stm;

   stm << "L" << theLayer;

   if ( theLadder < 0) stm << "-"; else stm <<"+";
   stm << abs(theLadder); 
   if (isFullModule() ) stm <<"F"; else stm <<"H";

   stm << "Z";
   if ( theLadder < 0) stm <<"-"; else stm <<"+";
   stm << abs(theModule);

   return stm.str();
}

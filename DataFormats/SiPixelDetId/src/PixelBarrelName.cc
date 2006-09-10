#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

#include <sstream>
#include <iostream>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

using namespace std;


PixelBarrelName::PixelBarrelName(const DetId & id) 
  : PixelModuleName(true)
{
 
//  uint32_t rawId = id.rawId(); 
  PXBDetId cmssw_numbering(id);


  theLayer = cmssw_numbering.layer();

  int oldModule = cmssw_numbering.module() -4; if (oldModule<=0) oldModule--;

  int oldLadder = cmssw_numbering.ladder();
  if (theLayer == 1) {
    if (oldLadder <= 5) oldLadder = 6-oldLadder;
    else if (oldLadder >= 6 && oldLadder <= 15 ) oldLadder = 5-oldLadder;
    else if (oldLadder >= 16) oldLadder = 26-oldLadder;
  } 
  else if (theLayer == 2) {
    if (oldLadder <= 8) oldLadder = 9-oldLadder;
    else if (oldLadder >= 9 && oldLadder <= 24) oldLadder = 8-oldLadder;
    else if (oldLadder >= 25) oldLadder = 41-oldLadder; 
  } 
  else if (theLayer == 3) {
    if (oldLadder <= 11) oldLadder = 12-oldLadder;
    else if (oldLadder >= 12 && oldLadder <= 33) oldLadder = 11-oldLadder;
    else if (oldLadder >= 34) oldLadder = 56-oldLadder;
  } 

  //
  // part
  //
  if      (oldModule < 0 && oldLadder < 0) thePart = mO; 
  else if (oldModule > 0 && oldLadder < 0) thePart = pO;
  else if (oldModule < 0 && oldLadder > 0) thePart = mI;
  else if (oldModule > 0 && oldLadder > 0) thePart = pI;
  

  //
  // ladder
  //
  theLadder = abs(oldLadder);

  //
  // module
  //
  theModule = abs(oldModule);
 
  //
  // half or full - this is a question...
  //

  halfModule = false; 
  if (theLadder == 1) halfModule = true;
  if (theLayer == 1 && theLadder == 10) halfModule = true;
  if (theLayer == 2 && theLadder == 16) halfModule = true;
  if (theLayer == 3 && theLadder == 22) halfModule = true;

  //
  // sector
  //
  theSector = 0;   // temporary
  if (theLayer==1) {
    switch (theLadder) {
    case 1 : case 2: {theSector = 1; break;}
    case 3 :         {theSector = 2; break;}
    case 4 :         {theSector = 3; break;}
    case 5 :         {theSector = 4; break;}
    case 6 :         {theSector = 5; break;}
    case 7 :         {theSector = 6; break;}
    case 8 :         {theSector = 7; break;}
    case 9 : case 10:{theSector = 8; break;}
    default: ;
    };
  } else if (theLayer==2) {
    switch (theLadder) {
    case  1 : case  2: {theSector = 1; break;}
    case  3 : case  4: {theSector = 2; break;}
    case  5 : case  6: {theSector = 3; break;}
    case  7 : case  8: {theSector = 4; break;}
    case  9 : case 10: {theSector = 5; break;}
    case 11 : case 12: {theSector = 6; break;}
    case 13 : case 14: {theSector = 7; break;}
    case 15 : case 16: {theSector = 8; break;}
    default: ;
    };
  } else if (theLayer==3) {
    switch (theLadder) {
    case  1 : case  2: case  3: {theSector = 1; break;}
    case  4 : case  5: case  6: {theSector = 2; break;}
    case  7 : case  8: case  9: {theSector = 3; break;}
    case 10 : case 11:          {theSector = 4; break;}
    case 12 : case 13:          {theSector = 5; break;}
    case 14 : case 15: case 16: {theSector = 6; break;}
    case 17 : case 18: case 19: {theSector = 7; break;}
    case 20 : case 21: case 22: {theSector = 8; break;}
    default: ;
    };
  }
}

string PixelBarrelName::name() const 
{
   std::ostringstream stm;
   
   stm<<"BPix_B:"<<thePart<<"_SEC:"<<theSector<<"_LYR:"<<theLayer<<"_LDR:"<<theLadder;
   if (halfModule) stm <<"H"; else stm <<"F";
   stm << "_MOD:" << theModule;

   return stm.str();
}

std::ostream & operator<<( std::ostream& out, const PixelBarrelName::Shell& t)
{
  switch (t) {
    case(PixelBarrelName::pI) : {out << "pI"; break;}
    case(PixelBarrelName::pO) : {out << "pO"; break;}
    case(PixelBarrelName::mI) : {out << "mI"; break;}
    case(PixelBarrelName::mO) : {out << "mO"; break;}
    default: out << "unknown";
  };
  return out;
}

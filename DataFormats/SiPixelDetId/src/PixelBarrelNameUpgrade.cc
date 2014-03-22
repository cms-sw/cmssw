#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"

#include <sstream>
#include <iostream>

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

PixelBarrelNameUpgrade::PixelBarrelNameUpgrade(const DetId & id) 
  : PixelModuleName(true)
{
 
//  uint32_t rawId = id.rawId(); 
  PXBDetId cmssw_numbering(id);


  theLayer = cmssw_numbering.layer();

  int oldModule = cmssw_numbering.module() -4; if (oldModule<=0) oldModule--;	// -4, ..., -1, +1, ...,+4

  int oldLadder = cmssw_numbering.ladder();

  if (theLayer == 1) {
    if (oldLadder <= 3) oldLadder = 4-oldLadder;				// +1, ..., +3
    else if (oldLadder >= 4 && oldLadder <= 9 ) oldLadder = 3-oldLadder;	// -1, ..., -6
    else if (oldLadder >= 10) oldLadder = 16-oldLadder;				// +6, ..., +4
  } 
  else if (theLayer == 2) {
    if (oldLadder <= 7) oldLadder = 8-oldLadder;
    else if (oldLadder >= 8 && oldLadder <= 21) oldLadder = 7-oldLadder;
    else if (oldLadder >= 22) oldLadder = 36-oldLadder; 
  } 
  else if (theLayer == 3) {
    if (oldLadder <= 11) oldLadder = 12-oldLadder;
    else if (oldLadder >= 12 && oldLadder <= 33) oldLadder = 11-oldLadder;
    else if (oldLadder >= 34) oldLadder = 56-oldLadder;
  } 
  else if (theLayer == 4) {
    if (oldLadder <= 16) oldLadder = 17-oldLadder;
    else if (oldLadder >= 17 && oldLadder <= 48) oldLadder = 16-oldLadder;
    else if (oldLadder >= 49) oldLadder = 81-oldLadder;
  } 
  
  //
  // part
  //
  if      (oldModule < 0 && oldLadder < 0) thePart = mO; 
  else if (oldModule > 0 && oldLadder < 0) thePart = pO;  
  else if (oldModule < 0 && oldLadder > 0) thePart = mI; 
  else if (oldModule > 0 && oldLadder > 0) thePart = pI; 
  //std::cout << "(oldModule, oldLadder)============(" << oldModule << ", "<< oldLadder<<" )" << std::endl;
  
  //
  // ladder
  //
  theLadder = abs(oldLadder);

  //
  // module
  //
  theModule = abs(oldModule);
 
}


// constructor from name string
PixelBarrelNameUpgrade::PixelBarrelNameUpgrade(std::string name) 
  : PixelModuleName(true), thePart(mO), theLayer(0),
    theModule(0), theLadder(0) {
  std::cout<<"NAME="<<name<<std::endl;
  // parse the name string
  // first, check to make sure this is an BPix name, should start with "BPix_"
  // also check to make sure the needed parts are present
  if ( (name.substr(0, 5) != "BPix_") ||
       (name.find("_B") == string::npos) || 
       (name.find("_LYR") == string::npos) ||
       (name.find("_LDR") == string::npos) || 
       (name.find("_MOD") == string::npos) ) {
    edm::LogError ("BadNameString|SiPixel") 
      << "Bad name string in PixelBarrelNameUpgrade::PixelBarrelName(std::string): "
      << name;
    return;
  }

  // strip off ROC part if it's there
  if (name.find("_ROC") != string::npos)
    name = name.substr(0, name.find("_ROC"));

  // find shell
  string shellString = name.substr(name.find("_B")+2, name.find("_SEC")-name.find("_B")-2);
  if (shellString == "mO") thePart = mO;
  else if (shellString == "mI") thePart = mI;
  else if (shellString == "pO") thePart = pO;
  else if (shellString == "pI") thePart = pI;
  else {
    edm::LogError ("BadNameString|SiPixel")
      << "Unable to determine shell in PixelBarrelNameUpgrade::PixelBarrelName(std::string): "
      << name;
  }

  // find the layer
  string layerString = name.substr(name.find("_LYR")+4, name.find("_LDR")-name.find("_LYR")-4);
  if (layerString == "1") theLayer = 1;
  else if (layerString == "2") theLayer = 2;
  else if (layerString == "3") theLayer = 3;
  else if (layerString == "4") theLayer = 4;
  else {
    edm::LogError ("BadNameString|SiPixel")
      << "Unable to determine layer in PixelBarrelNameUpgrade::PixelBarrelName(std::string): "
      << name;
  }
  
  // find the ladder
  string ladderString = name.substr(name.find("_LDR")+4, name.find("_MOD")-name.find("_LDR")-4);
  if (ladderString.substr(ladderString.size()-1, 1) == "F") {
    int ladderNum = atoi(ladderString.substr(0, ladderString.size() -1).c_str());
    if (theLayer == 1 && ladderNum >= 1 && ladderNum <= 6) theLadder = ladderNum;
    else if (theLayer == 2 && ladderNum >= 1 && ladderNum <= 14) theLadder = ladderNum;
    else if (theLayer == 3 && ladderNum >= 1 && ladderNum <= 22) theLadder = ladderNum;
    else if (theLayer == 4 && ladderNum >= 1 && ladderNum <= 32) theLadder = ladderNum;
    else {
      edm::LogError ("BadNameString|SiPixel")
	<< "Unable to determine ladder in PixelBarrelNameUpgrade::PixelBarrelName(std::string): "
	<< name;
    }
  } // full ladders
  else {
    edm::LogError ("BadNameString|SiPixel")
      << "Unable to determine ladder in PixelBarrelNameUpgrade::PixelBarrelName(std::string): "
      << name;
  }

  // find the module
  string moduleString = name.substr(name.find("_MOD")+4, name.size()-name.find("_MOD")-4);
  if (moduleString == "1") theModule = 1;
  else if (moduleString == "2") theModule = 2;
  else if (moduleString == "3") theModule = 3;
  else if (moduleString == "4") theModule = 4;
  else {
    edm::LogError ("BadNameString|SiPixel")
      << "Unable to determine module in PixelBarrelNameUpgrade::PixelBarrelName(std::string): "
      << name;
  }

} // PixelBarrelNameUpgrade::PixelBarrelName(std::string name)


int PixelBarrelNameUpgrade::sectorName() const
{
  int sector = 0;
  if (theLayer==1) {
    switch (theLadder) {
    case 1 :  {sector = 1; break;}
    case 2 :  {sector = 2; break;}
    case 3 :  {sector = 3; break;}
    case 4 :  {sector = 6; break;}
    case 5 :  {sector = 7; break;}
    case 6 :  {sector = 8; break;}
    default: ;
    };
  } else if (theLayer==2) {
    switch (theLadder) {
    case  1 : case  2: {sector = 1; break;}
    case  3 : case  4: {sector = 2; break;}
    case  5 : case  6: {sector = 3; break;}
    case  7 :          {sector = 4; break;}
    case  8 : 	       {sector = 5; break;}
    case  9 : case 10: {sector = 6; break;}
    case 11 : case 12: {sector = 7; break;}
    case 13 : case 14: {sector = 8; break;}    
    default: ;
    };
  } else if (theLayer==3) {
    switch (theLadder) {
    case  1 : case  2: case  3: {sector = 1; break;}
    case  4 : case  5: case  6: {sector = 2; break;}
    case  7 : case  8: case  9: {sector = 3; break;}
    case 10 : case 11:          {sector = 4; break;}
    case 12 : case 13:          {sector = 5; break;}
    case 14 : case 15: case 16: {sector = 6; break;}
    case 17 : case 18: case 19: {sector = 7; break;}
    case 20 : case 21: case 22: {sector = 8; break;}
    default: ;
    };
  }
   else if (theLayer==4) {
    switch (theLadder) {
    case  1 : case  2: case  3:  case  4: {sector = 1; break;}
    case  5 : case  6: case  7:  case  8: {sector = 2; break;}
    case  9 : case 10: case 11:  case 12: {sector = 3; break;}
    case 13 : case 14: case 15:  case 16: {sector = 4; break;}
    case 17 : case 18: case 19:  case 20: {sector = 5; break;}
    case 21 : case 22: case 23:  case 24: {sector = 6; break;}
    case 25 : case 26: case 27:  case 28: {sector = 7; break;}
    case 29 : case 30: case 31:  case 32: {sector = 8; break;}
    default: ;
    };
  }
  
  
  return sector;

}

//---- this needs to be removed --------------------------------
//---- there is no half module in the upgraded geometry!!! -----
bool PixelBarrelNameUpgrade::isHalfModule() const 
{
  bool halfModule = false;
//  if (theLadder == 1) halfModule = true;
//  if (theLayer == 1 && theLadder == 10) halfModule = true;
//  if (theLayer == 2 && theLadder == 16) halfModule = true;
//  if (theLayer == 3 && theLadder == 22) halfModule = true;
  return halfModule;
}


PixelModuleName::ModuleType  PixelBarrelNameUpgrade::moduleType() const 
{
  return isHalfModule() ? PixelBarrelNameUpgrade::v1x8 : PixelBarrelNameUpgrade::v2x8;
}



bool PixelBarrelNameUpgrade::operator==(const PixelModuleName &o) const
{
  if ( o.isBarrel() ) { 
    const PixelBarrelNameUpgrade *other = dynamic_cast<const PixelBarrelNameUpgrade*>(&o);
    return (    other
             && thePart   == other->thePart 
             && theLayer  == other->theLayer 
             && theModule == other->theModule 
             && theLadder == other->theLadder);
  } else return false; 
}


string PixelBarrelNameUpgrade::name() const 
{
   std::ostringstream stm;
   
   stm<<"BPix_B"<<thePart<<"_SEC"<<sectorName()<<"_LYR"<<theLayer<<"_LDR"<<theLadder;
   stm <<"F";
   stm << "_MOD" << theModule;
   
   return stm.str();
}
/*
string PixelBarrelNameUpgrade::name() const 
{
   std::ostringstream stm;
   
   stm<<"BPix_B"<<thePart<<"_SEC"<<sectorName()<<"_LYR"<<theLayer<<"_LDR"<<theLadder;
   if ( isHalfModule() ) stm <<"H"; else stm <<"F";
   stm << "_MOD" << theModule;

   return stm.str();
}
*/

// return the DetId
PXBDetId PixelBarrelNameUpgrade::getDetId() {
  
  uint32_t layer = 0;
  uint32_t ladder = 0;
  uint32_t module = 0;

  layer = layerName();
  uint32_t tmpLadder = ladderName();
  uint32_t tmpModule = moduleName();

  // translate the ladder number from the naming convention to the cmssw convention
  bool outer = false;
  Shell shell = thePart;
  outer = (shell == mO) || (shell == pO);
  if (outer) {
    if (layer == 1)
      ladder = tmpLadder + 3;
    else if (layer == 2)
      ladder = tmpLadder + 7;
    else if (layer == 3)
      ladder = tmpLadder + 11;
    else if (layer == 4)
      ladder = tmpLadder + 16;
  } // outer
  else { // inner
    if (layer == 1) {
      if (tmpLadder <= 3) ladder = 4 - tmpLadder;
      else if (tmpLadder <= 6) ladder = 16 - tmpLadder;
    } // layer 1
    else if (layer == 2) {
      if (tmpLadder <= 7) ladder = 8 - tmpLadder;
      else if (tmpLadder <= 14) ladder = 36 - tmpLadder;
    } // layer 2
    else if (layer == 3) {
      if (tmpLadder <= 11) ladder = 12 - tmpLadder;
      else if (tmpLadder <= 22) ladder = 56 - tmpLadder;
    } // layer 3
    else if (layer == 4) {
      if (tmpLadder <= 16) ladder = 17 - tmpLadder;
      else if (tmpLadder <= 32) ladder = 81 - tmpLadder;
    } // layer 4
  } // inner

  // translate the module number from naming convention to cmssw convention
  // numbering starts at positive z
  if (shell == pO || shell == pI)
    module = tmpModule + 4;
  else // negative z side
    module = 5 - tmpModule;

  return PXBDetId(layer, ladder, module);

} // PXBDetId PixelBarrelNameUpgrade::getDetId()



std::ostream & operator<<( std::ostream& out, const PixelBarrelNameUpgrade::Shell& t)
{
  switch (t) {
    case(PixelBarrelNameUpgrade::pI) : {out << "pI"; break;}
    case(PixelBarrelNameUpgrade::pO) : {out << "pO"; break;}
    case(PixelBarrelNameUpgrade::mI) : {out << "mI"; break;}
    case(PixelBarrelNameUpgrade::mO) : {out << "mO"; break;}
    default: out << "unknown";
  };
  return out;
}



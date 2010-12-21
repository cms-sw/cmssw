//----------------------------------------------------------------------
//
//   Class: DTConfigManager
//
//   Description: DT Configuration manager includes config classes for every single chip
//
//
//   Author List:
//   C.Battilana
//
//   april 07 : SV DTConfigTrigUnit added
//   april 07 : CB Removed DTGeometry dependecies
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

using namespace std;

//----------------
// Constructors --
//----------------

DTConfigManager::DTConfigManager(){

}
	
//--------------
// Destructor --
//--------------

DTConfigManager::~DTConfigManager(){

  my_sectcollmap.clear();
  my_trigunitmap.clear();
  my_tsphimap.clear();
  my_tsthetamap.clear();
  my_tracomap.clear();
  my_btimap.clear();

}

//--------------
// Operations --
//--------------

DTConfigBti* DTConfigManager::getDTConfigBti(DTBtiId btiid) const {
  
  DTChamberId chambid = btiid.SLId().chamberId();
  BtiMap::const_iterator biter1 = my_btimap.find(chambid);
  if (biter1 == my_btimap.end()){
    std::cout << "DTConfigManager::getConfigBti : Chamber (" << chambid.wheel()
	      << "," << chambid.sector()
	      << "," << chambid.station() 
	      << ") not found, return 0" << std::endl;
    return 0;
  }
  
  innerBtiMap::const_iterator biter2 = (*biter1).second.find(btiid);
  if (biter2 == (*biter1).second.end()){
    std::cout << "DTConfigManager::getConfigBti : BTI (" << btiid.wheel()
	      << "," << btiid.sector()
	      << "," << btiid.station()
	      << "," << btiid.superlayer()
	      << "," << btiid.bti()
	      << ") not found, return 0" << std::endl;
    return 0;
  }
  return const_cast<DTConfigBti*>(&(*biter2).second);

}  

const std::map<DTBtiId,DTConfigBti>& DTConfigManager::getDTConfigBtiMap(DTChamberId chambid) const {
  
  BtiMap::const_iterator biter = my_btimap.find(chambid);
  if (biter == my_btimap.end()){
    std::cout << "DTConfigManager::getConfigBtiMap : Chamber (" << chambid.wheel()
	      << "," << chambid.sector()
	      << "," << chambid.station() 
	      << ") not found, return a reference to the end of the map" << std::endl;
  }
  
  return (*biter).second;

}

DTConfigTraco* DTConfigManager::getDTConfigTraco(DTTracoId tracoid) const {
  
  DTChamberId chambid = tracoid.ChamberId();
  TracoMap::const_iterator titer1 = my_tracomap.find(chambid);
  if (titer1 == my_tracomap.end()){
    std::cout << "DTConfigManager::getConfigTraco : Chamber (" << chambid.wheel()
	      << "," << chambid.sector()
	      << "," << chambid.station() 
	      << ") not found, return 0" << std::endl;
    return 0;
  }
  
  innerTracoMap::const_iterator titer2 = (*titer1).second.find(tracoid);
  if (titer2 == (*titer1).second.end()){
    std::cout << "DTConfigManager::getConfigTraco : TRACO (" << tracoid.wheel()
	      << "," << tracoid.sector()
	      << "," << tracoid.station()
	      << "," << tracoid.traco()
	      << ") not found, return a reference to the end of the map" << std::endl;
    return 0;
  }
  return const_cast<DTConfigTraco*>(&(*titer2).second);

}

const std::map<DTTracoId,DTConfigTraco>& DTConfigManager::getDTConfigTracoMap(DTChamberId chambid) const {
  
  TracoMap::const_iterator titer = my_tracomap.find(chambid);
  if (titer == my_tracomap.end()){
    std::cout << "DTConfigManager::getConfigTracoMap : Chamber (" << chambid.wheel()
	      << "," << chambid.sector()
	      << "," << chambid.station() 
	      << ") not found, return 0" << std::endl;
  }

  return (*titer).second;

}

DTConfigTSTheta* DTConfigManager::getDTConfigTSTheta(DTChamberId chambid) const {
 
  TSThetaMap::const_iterator thiter = my_tsthetamap.find(chambid);
  if (thiter == my_tsthetamap.end()){
    std::cout << "DTConfigManager::getConfigTSTheta : Chamber (" << chambid.wheel()
	      << "," << chambid.sector()
	      << "," << chambid.station() 
	      << ") not found, return 0" << std::endl;
    return 0;
  }
  
  return const_cast<DTConfigTSTheta*>(&(*thiter).second);

}

DTConfigTSPhi* DTConfigManager::getDTConfigTSPhi(DTChamberId chambid) const {
  
  TSPhiMap::const_iterator phiter = my_tsphimap.find(chambid);
  if (phiter == my_tsphimap.end()){
    std::cout << "DTConfigManager::getConfigTSPhi : Chamber (" << chambid.wheel()
	      << "," << chambid.sector()
	      << "," << chambid.station() 
	      << ") not found, return 0" << std::endl;
    return 0;
  }

  return const_cast<DTConfigTSPhi*>(&(*phiter).second);

}
  
DTConfigTrigUnit* DTConfigManager::getDTConfigTrigUnit(DTChamberId chambid) const {
  
   TrigUnitMap::const_iterator tuiter = my_trigunitmap.find(chambid);
   if (tuiter == my_trigunitmap.end()){
     std::cout << "DTConfigManager::getConfigTrigUnit : Chamber (" << chambid.wheel()
 	      << "," << chambid.sector()
 	      << "," << chambid.station() 
 	      << ") not found, return 0" << std::endl;
     return 0;
   }

   return const_cast<DTConfigTrigUnit*>(&(*tuiter).second);

}

DTConfigLUTs* DTConfigManager::getDTConfigLUTs(DTChamberId chambid) const {
  
   LUTMap::const_iterator lutiter = my_lutmap.find(chambid);
   if (lutiter == my_lutmap.end()){
     std::cout << "DTConfigManager::getConfigLUTs : Chamber (" << chambid.wheel()
 	      << "," << chambid.sector()
 	      << "," << chambid.station() 
 	      << ") not found, return 0" << std::endl;
     return 0;
   }

   return const_cast<DTConfigLUTs*>(&(*lutiter).second);

}

DTConfigSectColl* DTConfigManager::getDTConfigSectColl(DTSectCollId scid) const {
  
  SectCollMap::const_iterator sciter = my_sectcollmap.find(scid);
  if (sciter == my_sectcollmap.end()){
    std::cout << "DTConfigManager::getConfigSectColl : SectorCollector (" << scid.wheel()
	      << "," << scid.sector() 
	      << ") not found, return 0" << std::endl;
    return 0;
  }

  return const_cast<DTConfigSectColl*>(&(*sciter).second);

}

void DTConfigManager::setDTConfigBti(DTBtiId btiid,DTConfigBti conf){

  DTChamberId chambid = btiid.SLId().chamberId();
  my_btimap[chambid][btiid] = conf;

}  

void DTConfigManager::setDTConfigTraco(DTTracoId tracoid,DTConfigTraco conf){

  DTChamberId chambid = tracoid.ChamberId();
  my_tracomap[chambid][tracoid] = conf;

}  

int DTConfigManager::getBXOffset() const {

  int ST = static_cast<int>(getDTConfigBti(DTBtiId(1,1,1,1,1))->ST());
  int coarse = getDTConfigSectColl(DTSectCollId(1,1))->CoarseSync(1);
  return (ST/2 + ST%2 + coarse); //CB check this function!

}


void DTConfigManager::dumpLUTParam(DTChamberId &chambid) const {

  // open txt file
  string name = "Lut_from_param";
  name += ".txt";

  ofstream fout;
  fout.open(name.c_str(),ofstream::app);

  // get wheel, station, sector from chamber 
  int wh = chambid.wheel();
  int st = chambid.station();
  int se = chambid.sector();

  //cout << "Dumping lut command for wh " << wh << " st " << st << " se " << se << endl;

  fout << wh;
  fout << "\t" << st;
  fout << "\t" << se;

  // get parameters from configuration
  // get DTConfigLUTs for this chamber
  DTConfigLUTs* _confLUTs = getDTConfigLUTs(chambid);
  short int btic = getDTConfigTraco(DTTracoId(wh,st,se,1))->BTIC(); 
  float d = _confLUTs->D();
  float xcn = _confLUTs->Xcn();
  //fout << "\td\t" << d << "\txcn\t" << xcn << "\t"; 
  //fout << "btic\t" << btic << "\t";

  // *** dump TRACO LUT command
  fout << "\tA8";
  short int Low_byte = (btic & 0x00FF);   // output in hex bytes format with zero padding
  short int High_byte =( btic>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0') << Low_byte;

  // convert parameters from IEE32 float to DSP float format
  short int DSPmantissa = 0;
  short int DSPexp = 0;

  // d parameter conversion and dump
  _confLUTs->IEEE32toDSP(d, DSPmantissa, DSPexp);
  Low_byte = (DSPmantissa & 0x00FF);   // output in hex bytes format with zero padding
  High_byte =( DSPmantissa>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0') << Low_byte;
  Low_byte = (DSPexp & 0x00FF);
  High_byte =( DSPexp>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << High_byte << setw(2) << setfill('0') << Low_byte;

  // xnc parameter conversion and dump
  DSPmantissa = 0;
  DSPexp = 0;
  _confLUTs->IEEE32toDSP(xcn, DSPmantissa, DSPexp);
  Low_byte = (DSPmantissa & 0x00FF);   // output in hex bytes format with zero padding
  High_byte =( DSPmantissa>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0') << Low_byte;
  Low_byte = (DSPexp & 0x00FF);
  High_byte =( DSPexp>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << High_byte << setw(2) << setfill('0') << Low_byte;

  // sign bits
  short int xcn_sign = _confLUTs->Wheel();
  Low_byte = (xcn_sign & 0x00FF);   // output in hex bytes format with zero padding
  High_byte =( xcn_sign>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0') << Low_byte << dec << "\n";

  fout.close();

  return;

}

//----------------------------------------------------------------------
//
//   Class: DTConfigManager
//
//   Description: DT Configuration manager includes config classes for every single chip
//
//
//   Author List:
//   C.Battilana
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
               
//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

using namespace std;
//----------------
// Constructors --
//----------------

DTConfigManager::DTConfigManager(edm::ParameterSet& ps, edm::ESHandle<DTGeometry>& gha){

  //create config classes&C.
  my_dttpgdebug = ps.getUntrackedParameter<bool>("Debug");
  //my_dttpgdebug = true;

  my_sectcollconf = new DTConfigSectColl(ps.getParameter<edm::ParameterSet>("SectCollParameters"));
  edm::ParameterSet tups = ps.getParameter<edm::ParameterSet>("TUParameters");
  my_trigunitdebug = tups.getUntrackedParameter<bool>("Debug");
  my_bticonf = new DTConfigBti(tups.getParameter<edm::ParameterSet>("BtiParameters"));
  my_tracoconf = new DTConfigTraco(tups.getParameter<edm::ParameterSet>("TracoParameters"));
  my_tsthetaconf = new DTConfigTSTheta(tups.getParameter<edm::ParameterSet>("TSThetaParameters"));
  my_tsphiconf = new DTConfigTSPhi(tups.getParameter<edm::ParameterSet>("TSPhiParameters"));
  
  // loop to create maps
  for (std::vector<DTChamber*>::const_iterator ich=gha->chambers().begin(); ich!=gha->chambers().end();++ich){
    DTChamberId   chambid = (*ich)->id();
    innerBtiMap   &ibtimap = my_btimap[chambid] ;
    innerTracoMap &itracomap = my_tracomap[chambid] ;

    if(my_dttpgdebug)
    {
    	std::cout << " Filling configuration for chamber : wh " << chambid.wheel() << 
		", st " << chambid.station() << 
		", se " << chambid.sector() << endl;
    }

    //fill the bti map
    for (int isl=1;isl<=3;isl++){ //CB check x la velocità di rimepimento di sto coso!
      int ncell = 0;
      if( chambid.station()==4 && isl==2){
	ncell = 0;
      }
      else {
	ncell = (*ich)->layer(DTLayerId(chambid,isl,1))->specificTopology().channels() + 1;
      }
      for (int ibti=0;ibti<ncell;ibti++)
      {
	ibtimap[DTBtiId(chambid,isl,ibti+1)] = my_bticonf;
        if(my_dttpgdebug)
		std::cout << "Filling BTI config for chamber : wh " << chambid.wheel() << 
		", st " << chambid.station() << 
		", se " << chambid.sector() << 
		"... sl " << isl << 
		", bti " << ibti+1 << endl;
      }     
    }
    
    // fill the traco map
    int ntraco = ((*ich)->layer(DTLayerId(chambid,1,1))->specificTopology().channels()/(DTConfig::NBTITC))+1;
    for (int itraco=0;itraco<ntraco;itraco++)
    { 
      itracomap[DTTracoId(chambid,itraco+1)] = my_tracoconf;
      if(my_dttpgdebug)
		std::cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << 
		", st " << chambid.station() << 
		", se " << chambid.sector() << 
		", traco " << itraco+1 << endl;
    }     
    
  //   std::cout <<"BtiMap & TracoMap Size for chamber : wh " << chambid.wheel() << 
// 		", st " << chambid.station() << 
// 		", se " << chambid.sector() << 
// 		" ---> " << getDTConfigBtiMap(chambid).size() << " " << getDTConfigTracoMap(chambid).size() << endl;

    // fill TS map
    my_tsthetamap[chambid] = my_tsthetaconf;
    my_tsphimap[chambid] = my_tsphiconf;
  }
  
  //loop on Sector Collectors
  for (int wh=-2;wh<=2;wh++)
    for (int se=1;se<=12;se++)
      my_sectcollmap[DTSectCollId(wh,se)] = my_sectcollconf;

}
	
//--------------
// Destructor --
//--------------

DTConfigManager::~DTConfigManager(){

  my_sectcollmap.clear();
  my_tsphimap.clear();
  my_tsthetamap.clear();
  my_tracomap.clear();
  my_btimap.clear();

  delete my_bticonf;
  delete my_tracoconf;
  delete my_tsthetaconf;
  delete my_tsphiconf;
  delete my_sectcollconf;

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
  return (*biter2).second;

}  

const std::map<DTBtiId,DTConfigBti* >& DTConfigManager::getDTConfigBtiMap(DTChamberId chambid) const {
  
  BtiMap::const_iterator biter = my_btimap.find(chambid);
  if (biter == my_btimap.end()){
    std::cout << "DTConfigManager::getConfigBtiMap : Chamber (" << chambid.wheel()
	      << "," << chambid.sector()
	      << "," << chambid.station() 
	      << ") not found, return a reference to the end of the map" << std::endl;
    //return (*(my_btimap.end()).second); vedi cosa fare con questo!
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
  return (*titer2).second;

}

const std::map<DTTracoId,DTConfigTraco* >& DTConfigManager::getDTConfigTracoMap(DTChamberId chambid) const {
  
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
  
  return (*thiter).second;

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

  return (*phiter).second;

}

// DTConfigTrigUnit DTConfigManager::getConfigTrigUnit(DTChamberId chambid) const {
  
//   std::map<DTChamberId,DTConfigTrigUnit* >::iterator tuiter = my_tumap.find(chambid);
//   if (tuiter == my_tumap.end()){
//     std::cout << "DTCOnfigManager::getConfigTrigUnit : Chamber (" << chambid.wheel()
// 	      << "," << chambid.sector()
// 	      << "," << chambid.station() 
// 	      << ") not found, return 0" std::endl;
//     return 0;
//   }

//   return (*tuiter).second;

// }

DTConfigSectColl* DTConfigManager::getDTConfigSectColl(DTSectCollId scid) const {
  
  SectCollMap::const_iterator sciter = my_sectcollmap.find(scid);
  if (sciter == my_sectcollmap.end()){
    std::cout << "DTConfigManager::getConfigSectColl : SectorCollector (" << scid.wheel()
	      << "," << scid.sector() 
	      << ") not found, return 0" << std::endl;
    return 0;
  }

  return (*sciter).second;

}

int DTConfigManager::getBXOffset() const {

  int ST = static_cast<int>(getDTConfigBti(DTBtiId(1,1,1,1,1))->ST());
  int coarse = getDTConfigSectColl(DTSectCollId(1,1))->CoarseSync(1);
  return (ST/2 + ST%2 + coarse); //CB testalo!!!

}

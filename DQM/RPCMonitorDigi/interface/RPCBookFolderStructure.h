/** \class RPCBookFolderStructure
 *
 *  $Date: 2011/03/02 16:45:36 $
 *  $Revision: 1.5 $
 * \author Anna Cimmino (INFN Napoli)
 *
 * Create folder structure for DQM histo saving
 */
#ifndef RPCBookFolderStructure_H
#define RPCBookFolderStructure_H

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include<string>
#include <sstream>
class RPCBookFolderStructure {
   public:
      explicit RPCBookFolderStructure() {};
      ~RPCBookFolderStructure(){};

      std::string folderStructure(RPCDetId detId){ 
	
	std::stringstream myStream ;
	myStream.str("");

	if(detId.region() ==  0) 
	  myStream <<"Barrel/Wheel_"<<detId.ring()<<"/sector_"<<detId.sector()<<"/station_"<<detId.station();
	else if(detId.region() == -1) 
	  myStream <<"Endcap-/Disk_-"<<detId.station()<<"/ring_"<<detId.ring()<<"/sector_"<<detId.sector();
	else if(detId.region() ==  1) 
	  myStream <<"Endcap+/Disk_"<<detId.station()<<"/ring_"<<detId.ring()<<"/sector_"<<detId.sector();
	else  myStream <<"Error/Folder/Creation";

      return myStream.str();
      }
};
#endif

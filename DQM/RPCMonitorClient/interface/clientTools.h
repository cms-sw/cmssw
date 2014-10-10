#ifndef rpcdqmclient_clientTools_H
#define rpcdqmclient_clientTools_H

#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <vector>
#include <iomanip>


namespace rpcdqmclient{
  class clientTools{
  public:

  std::vector<MonitorElement*> constructMEVector(const edm::EventSetup& iSetup, const std::string & prefixDir, const std::string & MEName, DQMStore* dbe ){
    cout<<"Starting ConstructMEVector"<<endl;
    this->getMEs(iSetup, prefixDir,  MEName, dbe);

    return 	myMeVect_;
  }

  std::vector<RPCDetId>  getAssociatedRPCdetId(){
    std::vector<RPCDetId> myVector; 
    myVector.clear();

    if (myMeVect_.size() !=0 && myMeVect_.size()==myDetIds_.size() ) myVector= myDetIds_;

    return myVector;
  }


  protected:
 
  void getMEs(const edm::EventSetup& iSetup, const std::string & prefixDir, const std::string & MEName ,  DQMStore* dbe){


 
    edm::ESHandle<RPCGeometry> rpcGeo;
    iSetup.get<MuonGeometryRecord>().get(rpcGeo);
 
    //loop on all geometry and get all histos
    for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
      if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
 
	RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
	std::vector< const RPCRoll*> roles = (ch->rolls());
	//Loop on rolls in given chamber
	for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	  RPCDetId detId = (*r)->id();
	
	  //Get Occupancy ME for roll
	  RPCGeomServ RPCname(detId);	   
	  RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
	  MonitorElement * myMe = dbe->get(prefixDir+"/"+ folderStr->folderStructure(detId)+"/"+MEName+ "_"+RPCname.name()); 
	  if (!myMe)continue;

	  myMeVect_.push_back(myMe);
	  myDetIds_.push_back(detId);
	  myRollNames_.push_back(RPCname.name());

	}
      }
    }//end loop on all geometry and get all histos  
  }
  
 
  
  private:
  std::vector<MonitorElement *>  myMeVect_;
  std::vector<RPCDetId>   myDetIds_;
  std::vector<std::string>    myRollNames_;


  };
}

#endif

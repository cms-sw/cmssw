#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "DQM/RPCMonitorClient/interface/RPCDBHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "CondFormats/RPCObjects/interface/RPCDQMObject.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DQM/RPCMonitorDigi/interface/utils.h"
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"

//typedef popcon::PopConAnalyzer<RPCDBHandler> RPCDBPopConAnalyzer;

class RPCDBPopConAnalyzer: public popcon::PopConAnalyzer<RPCDBHandler>
{
public:
  typedef RPCDBHandler SourceHandler;

  RPCDBPopConAnalyzer(const edm::ParameterSet& pset): 
    popcon::PopConAnalyzer<RPCDBHandler>(pset),
    m_populator(pset),
    m_source(pset.getParameter<edm::ParameterSet>("Source")) {
      subsystemFolder_= pset.getUntrackedParameter<std::string>("RPCFolder", "RPC");
      recHitTypeFolder_= pset.getUntrackedParameter<std::string>("RecHitTypeFolder", "Noise");
      summaryFolder_= pset.getUntrackedParameter<std::string>("SummaryFolder", "SummaryHistograms");
      efficiencyFolder_= pset.getUntrackedParameter<std::string>("EfficiencyFolder", "RPCEfficiency");
    }

private:
  virtual void endJob() 
  {
    m_source.initObject(rpcDQMObject);
    write();
    dbe =0;
  }

  virtual void beginRun(const edm::Run& run, const edm::EventSetup& iSetup){
    dbe = edm::Service<DQMStore>().operator->();
    dbe->setCurrentFolder("RPCPVT");
  }//beginRun


  virtual void analyze(const edm::Event& ev, const edm::EventSetup& iSetup){ //}

  //virtual void endRun(const edm::Run& r, const edm::EventSetup& iSetup){

    rpcDQMObject = new RPCDQMObject();
    RPCDQMObject::DQMObjectItem rpcDqmItem;

    edm::ESHandle<RPCGeometry> rpcGeo;
    iSetup.get<MuonGeometryRecord>().get(rpcGeo);
    //Loop on RPC geometry to access ME for each roll

    RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure(); 
    rpcdqm::utils rpcUtils;

    for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
      if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
        RPCChamber* ch = dynamic_cast< RPCChamber* >( *it );
        std::vector< const RPCRoll*> roles = (ch->rolls());
        for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){

          //Get RPC DetId
          RPCDetId rpcId = (*r)->id();

	  //Check if rpcId is Valid. If not continue;

          //Get roll name
          RPCGeomServ rpcsrv(rpcId);
          std::string nameRoll = rpcsrv.name();

          //Get ME
          std::stringstream mepath;
          mepath.str("");
	  //          mepath<<"RPCPVT";
	  MonitorElement * MEnumcls = dbe->get(subsystemFolder_ + "/" + recHitTypeFolder_ + "/" + folderStr->folderStructure(rpcId)  + "/" + "NumberOfClusters_" + nameRoll);
          MonitorElement * MEcls = dbe->get(subsystemFolder_ + "/" + recHitTypeFolder_ + "/" + folderStr->folderStructure(rpcId)  + "/" + "ClusterSize_" + nameRoll);
          MonitorElement * MEocc= dbe->get(subsystemFolder_ + "/" + recHitTypeFolder_ + "/" + folderStr->folderStructure(rpcId) + "/" + "Occupancy_" + nameRoll);
          MonitorElement * MEbx = dbe->get(subsystemFolder_ + "/" + recHitTypeFolder_ + "/" + folderStr->folderStructure(rpcId) + "/" + "BXN_" + nameRoll);

          MonitorElement * MEstatus = NULL;
          MonitorElement * MEeff = NULL;
          rpcDqmItem.status = -999;
	  rpcDqmItem.efficiency= -999;

          if( rpcId.region() == 0){ //BARREL

            int nr = rpcUtils.detId2RollNr(rpcId);
            int sector = (int)rpcId.sector();

	    //Status
            mepath.str("");
            mepath<<subsystemFolder_<<"/" << recHitTypeFolder_<<"/" <<  summaryFolder_<<"/RPCChamberQuality_Roll_vs_Sector_Wheel"<<rpcId.ring();
            MEstatus = dbe->get(mepath.str());  
            if(MEstatus != 0 ){
              rpcDqmItem.status =  MEstatus->getBinContent(sector, nr);
            }else{
            edm::LogWarning("rpcdbclient")<< "[RPCDBClient] Did not find Status for Barrel "<< nameRoll;
	    }

	    //Efficiency
            mepath.str("");
            if( rpcId.ring() > 0){
              mepath<<subsystemFolder_<<"/" << efficiencyFolder_<<"/Efficiency_Roll_vs_Sector_Wheel_+"<<rpcId.ring();
            }else{
              mepath<<subsystemFolder_<<"/" << efficiencyFolder_<<"/Efficiency_Roll_vs_Sector_Wheel_"<<rpcId.ring();
            }

            MEeff = dbe->get(mepath.str());  
            if(MEeff != 0 ){
              rpcDqmItem.efficiency =  MEeff->getBinContent(sector, nr);
            }else{
            edm::LogWarning("rpcdbclient")<< "[RPCDBClient] Did not find Efficiency for Barrel "<< nameRoll;
	    }


          }else{
	    int segment =  rpcsrv.segment() ;
	    int endcapbin =  (rpcId.ring()-1)*3-rpcId.roll()+1;
	    int disk = (rpcId.region() * rpcId.layer());

	    //Status
            mepath.str("");
            mepath<<subsystemFolder_<<"/" << recHitTypeFolder_<<"/" <<  summaryFolder_<<"/RPCChamberQuality_Ring_vs_Segment_Disk"<<disk;
            MEstatus = dbe->get(mepath.str());
            if(MEstatus != 0 ){
              rpcDqmItem.status =   MEstatus->getBinContent(segment,endcapbin);
            }else{
            edm::LogWarning("rpcdbclient")<< "[RPCDBClient] Did not find Status for Endcap "<< nameRoll;
	    }


	    //Efficiency
            mepath.str("");
            mepath<<subsystemFolder_<<"/" << efficiencyFolder_<<"/Efficiency_Roll_vs_Segment_Disk_"<<disk;
            MEeff = dbe->get(mepath.str());  
            if(MEeff != 0 ){
              rpcDqmItem.efficiency =  MEeff->getBinContent(segment,endcapbin);
            }else{
            edm::LogWarning("rpcdbclient")<< "[RPCDBClient] Did not find Efficiency for Endcap "<< nameRoll;
	    }


          }
    
          rpcDqmItem.dpid = (int)rpcId;
	  rpcDqmItem.clusterSize = -999;
	  rpcDqmItem.numdigi = -999;
	    rpcDqmItem.numcluster =-999;
          rpcDqmItem.bx = -999;
          rpcDqmItem.bxrms = -999;
          //rpcDqmItem.status = -999;

	  if (MEnumcls != 0) {
            rpcDqmItem.numcluster = (float)MEnumcls->getMean();
          }else{
            edm::LogWarning("rpcdbclient")<< "[RPCDBClient] Did not find Number of Clusters for Roll "<< nameRoll;
          }

          if (MEcls != 0) {
            rpcDqmItem.clusterSize = (float)MEcls->getMean();
          }else{
            edm::LogWarning("rpcdbclient")<< "[RPCDBClient] Did not find ClusterSize for Roll "<< nameRoll;
          }

          if (MEbx != 0) {
            rpcDqmItem.bx = (float)MEbx->getMean();
            rpcDqmItem.bxrms = (float)MEbx->getRMS();
          }else{
            edm::LogWarning("rpcdbclient")<< "[RPCDBClient] Did not find BX for Roll "<< nameRoll;
          }
         
	  if (MEocc != 0) {
            rpcDqmItem.numdigi = (float)MEocc->getEntries();
          }else{
            edm::LogWarning("rpcdbclient")<< "[RPCDBClient] Did not find Occupancy for Roll "<< nameRoll;
          }
         


          (rpcDQMObject->v_cls).push_back(rpcDqmItem);
        }//End loop Rolls
      }
    }//End loop RPC Geometry

  }
  
  void write() { m_populator.write(m_source); }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;

  std::string subsystemFolder_;
  std::string summaryFolder_;
  std::string recHitTypeFolder_;
  std::string efficiencyFolder_;
  DQMStore * dbe;
  RPCDQMObject * rpcDQMObject;

};

DEFINE_FWK_MODULE(RPCDBPopConAnalyzer);


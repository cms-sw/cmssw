/** \class RPCEfficiencySecond
 * \Original author Camilo Carrillo (Uniandes)
 */

#include <DataFormats/MuonDetId/interface/RPCDetId.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include "FWCore/Framework/interface/ESHandle.h"

#include<string>
#include<map>
#include<fstream>

class RPCEfficiencySecond :public DQMEDHarvester{
   public:
      explicit RPCEfficiencySecond(const edm::ParameterSet&);
      ~RPCEfficiencySecond();
      int rollY(std::string shortname,const std::vector<std::string>& rollNames);
  
 protected:
  void beginJob();
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&); //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

 private:
  void myBooker(DQMStore::IBooker &);
      //Histograms to use
      MonitorElement * histoRPC;
      MonitorElement * histoDT;
      MonitorElement * histoRealRPC;
      MonitorElement * histoCSC;
           
      MonitorElement * histoPRO;
      MonitorElement * histoeffIdRPC_DT;
      MonitorElement * histoeffIdRPC_CSC;
      
      //For Duplication
      MonitorElement * histoRPC2;
      MonitorElement * histoDT2;
      MonitorElement * histoRealRPC2;
      MonitorElement * histoCSC2;
      //  MonitorElement * BXDistribution2;
      
      
      //Eff Global Barrel
      MonitorElement * EffGlobW[5];

      //Eff Distro Barrel
      MonitorElement * EffDistroW[5];

      //Eff Global EndCap      
      MonitorElement * EffGlobD[10];

      //EffDistro EndCap
      MonitorElement * EffDistroD[10];
      
      //Summary Histograms.
      MonitorElement * WheelSummary[5];
      MonitorElement * DiskSummary[10];
            
      //Azimultal Plots
      MonitorElement *  sectorEffW[5];        
      MonitorElement * OcsectorEffW[5];  
      MonitorElement * ExsectorEffW[5];  
      
      MonitorElement * GregR2D[10];  
      MonitorElement * GregR3D[10];  

      MonitorElement * OcGregR2D[10];  
      MonitorElement * OcGregR3D[10];  
      
      MonitorElement * ExGregR2D[10];  
      MonitorElement * ExGregR3D[10];  
      
      MonitorElement * ExpLayerW[5];      
      MonitorElement * ObsLayerW[5];

      edm::ESHandle<RPCGeometry> rpcGeo_;
	  
    

  std::map<int, std::map<std::string, MonitorElement*> >  meCollection;
  
  bool init_;
  
  std::string folderPath;
  int  numberOfDisks_;
  int innermostRings_ ;
    
};


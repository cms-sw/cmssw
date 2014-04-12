/** \class RPCEfficiencySecond
 * \Original author Camilo Carrillo (Uniandes)
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <DataFormats/MuonDetId/interface/RPCDetId.h>

#include<string>
#include<map>
#include<fstream>

class RPCDetId;


class RPCEfficiencySecond : public edm::EDAnalyzer {
   public:
      explicit RPCEfficiencySecond(const edm::ParameterSet&);
      ~RPCEfficiencySecond();
      int rollY(std::string shortname,const std::vector<std::string>& rollNames);
  
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

      
      
  
 private:
  virtual void beginRun(const edm::Run&, const edm::EventSetup& iSetup) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void endRun(const edm::Run& , const edm::EventSetup& );

  std::map<std::string, MonitorElement*> bookDetUnitSeg(RPCDetId & detId,int nstrips,std::string folder);
  std::map<int, std::map<std::string, MonitorElement*> >  meCollection;
  
  bool debug;
  bool SaveFile;
  std::string NameFile;
  std::string folderPath;
   int  numberOfDisks_;
  int innermostRings_ ;
  DQMStore * dbe;
  
};


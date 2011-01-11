/** \class RPCEfficiency
 *
 * Class for RPC Monitoring: uses RPCDigi and DT and CSC Segments.
 *
 *  $Date: 2010/09/20 14:39:10 $
 *  $Revision: 1.3 $
 *
 * \author Camilo Carrillo (Uniandes)
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include<string>
#include<map>
#include<fstream>

class RPCDetId;
class TFile;
class TH1F;
class TFile;
class TCanvas;
class TH2F;
class TString;


class DTStationIndex{
public: 
  DTStationIndex():_region(0),_wheel(0),_sector(0),_station(0){}
  DTStationIndex(int region, int wheel, int sector, int station) : 
    _region(region),
    _wheel(wheel),
    _sector(sector),
    _station(station){}
  ~DTStationIndex(){}
  int region() const {return _region;}
  int wheel() const {return _wheel;}
  int sector() const {return _sector;}
  int station() const {return _station;}
  bool operator<(const DTStationIndex& dtind) const{
    if(dtind.region()!=this->region())
      return dtind.region()<this->region();
    else if(dtind.wheel()!=this->wheel())
      return dtind.wheel()<this->wheel();
    else if(dtind.sector()!=this->sector())
      return dtind.sector()<this->sector();
    else if(dtind.station()!=this->station())
      return dtind.station()<this->station();
    return false;
  }
private:
  int _region;
  int _wheel;
  int _sector;
  int _station; 
};


class CSCStationIndex{
public:
  CSCStationIndex():_region(0),_station(0),_ring(0),_chamber(0){}
  CSCStationIndex(int region, int station, int ring, int chamber):
    _region(region),
    _station(station),
    _ring(ring),
    _chamber(chamber){}
  ~CSCStationIndex(){}
  int region() const {return _region;}
  int station() const {return _station;}
  int ring() const {return _ring;}
  int chamber() const {return _chamber;}
  bool operator<(const CSCStationIndex& cscind) const{
    if(cscind.region()!=this->region())
      return cscind.region()<this->region();
    else if(cscind.station()!=this->station())
      return cscind.station()<this->station();
    else if(cscind.ring()!=this->ring())
      return cscind.ring()<this->ring();
    else if(cscind.chamber()!=this->chamber())
      return cscind.chamber()<this->chamber();
    return false;
  }

private:
  int _region;
  int _station;
  int _ring;  
  int _chamber;
};

class RPCEfficiency : public edm::EDAnalyzer {
   public:
      explicit RPCEfficiency(const edm::ParameterSet&);
      ~RPCEfficiency();
      virtual void beginJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&);
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::map<std::string, MonitorElement*> bookDetUnitSeg(RPCDetId & detId,int nstrips, std::string folder);
      virtual void endRun(const edm::Run& r, const edm::EventSetup& iSetup);
      std::map<DTStationIndex,std::set<RPCDetId> > rollstoreDT;
      std::map<CSCStationIndex,std::set<RPCDetId> > rollstoreCSC;
      edm::ESHandle<RPCGeometry> rpcGeo;
      edm::ESHandle<DTGeometry> dtGeo;
      edm::ESHandle<CSCGeometry> cscGeo;

      MonitorElement * statistics;
      edm::InputTag RPCRecHitLabel_;
      //Distance Strip
      MonitorElement * DistBorderClu1La1;
      MonitorElement * DistBorderClu1La2;
      MonitorElement * DistBorderClu1La3;
      MonitorElement * DistBorderClu1La4;
      MonitorElement * DistBorderClu1La5;
      MonitorElement * DistBorderClu1La6;

      MonitorElement * DistBorderClu2La1;
      MonitorElement * DistBorderClu2La2;
      MonitorElement * DistBorderClu2La3;
      MonitorElement * DistBorderClu2La4;
      MonitorElement * DistBorderClu2La5;
      MonitorElement * DistBorderClu2La6;

      MonitorElement * DistBorderClu3La1;
      MonitorElement * DistBorderClu3La2;
      MonitorElement * DistBorderClu3La3;
      MonitorElement * DistBorderClu3La4;
      MonitorElement * DistBorderClu3La5;
      MonitorElement * DistBorderClu3La6;
      
      //Ang Dependence
      MonitorElement * AngClu1La1;
      MonitorElement * AngClu1La2;
      MonitorElement * AngClu1La3;
      MonitorElement * AngClu1La4;
      MonitorElement * AngClu1La5;
      MonitorElement * AngClu1La6;

      MonitorElement * AngClu2La1;
      MonitorElement * AngClu2La2;
      MonitorElement * AngClu2La3;
      MonitorElement * AngClu2La4;
      MonitorElement * AngClu2La5;
      MonitorElement * AngClu2La6;

      MonitorElement * AngClu3La1;
      MonitorElement * AngClu3La2;
      MonitorElement * AngClu3La3;
      MonitorElement * AngClu3La4;
      MonitorElement * AngClu3La5;
      MonitorElement * AngClu3La6;
      
      //Residuals

      MonitorElement * hGlobalResClu1La1;
      MonitorElement * hGlobalResClu1La2;
      MonitorElement * hGlobalResClu1La3;
      MonitorElement * hGlobalResClu1La4;
      MonitorElement * hGlobalResClu1La5;
      MonitorElement * hGlobalResClu1La6;

      MonitorElement * hGlobalResClu2La1;
      MonitorElement * hGlobalResClu2La2;
      MonitorElement * hGlobalResClu2La3;
      MonitorElement * hGlobalResClu2La4;
      MonitorElement * hGlobalResClu2La5;
      MonitorElement * hGlobalResClu2La6;

      MonitorElement * hGlobalResClu3La1;
      MonitorElement * hGlobalResClu3La2;
      MonitorElement * hGlobalResClu3La3;
      MonitorElement * hGlobalResClu3La4;
      MonitorElement * hGlobalResClu3La5;
      MonitorElement * hGlobalResClu3La6;

      //Endcap  
      MonitorElement * hGlobalResClu1R3C;
      MonitorElement * hGlobalResClu1R3B;
      MonitorElement * hGlobalResClu1R3A;
      MonitorElement * hGlobalResClu1R2C;
      MonitorElement * hGlobalResClu1R2B; 
      MonitorElement * hGlobalResClu1R2A;

      MonitorElement * hGlobalResClu2R3C;
      MonitorElement * hGlobalResClu2R3B;
      MonitorElement * hGlobalResClu2R3A;
      MonitorElement * hGlobalResClu2R2C;
      MonitorElement * hGlobalResClu2R2B;
      MonitorElement * hGlobalResClu2R2A;

      MonitorElement * hGlobalResClu3R3C;
      MonitorElement * hGlobalResClu3R3B;
      MonitorElement * hGlobalResClu3R3A;
      MonitorElement * hGlobalResClu3R2C;
      MonitorElement * hGlobalResClu3R2B;
      MonitorElement * hGlobalResClu3R2A;

 private:
      std::vector<std::map<RPCDetId, int> > counter;
      std::vector<int> totalcounter;
      std::ofstream ofrej;
      bool incldt;
      bool incldtMB4;
      bool inclcsc;
      bool debug;
      bool paper;
      bool inves;
      double rangestrips;
      double rangestripsRB4;
      double MinCosAng;
      double MaxD;
      double MaxDrb4;
      int dupli;
      std::string muonRPCDigis;
      std::string cscSegments;
      std::string dt4DSegments;
      std::string folderPath;
      std::string rejected;
      std::string rollseff;
      
      std::map<int, std::map<std::string, MonitorElement*> >  meCollection;
      
      bool EffSaveRootFile;
      std::string EffRootFileName;
      std::string nameInLog;

      DQMStore * dbe;
};

/** \class RPCEfficiency
 * Class for RPC Monitoring using RPCDigi and DT and CSC Segments.
 * \original author Camilo Carrillo (Uniandes)
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <DataFormats/MuonDetId/interface/RPCDetId.h>

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include<string>
#include<map>


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


class RPCEfficiency : public DQMEDAnalyzer {
   public:
      explicit RPCEfficiency(const edm::ParameterSet&);
      ~RPCEfficiency();


 protected:

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
      void bookDetUnitSeg(DQMStore::IBooker &, RPCDetId & detId,int nstrips, std::string folder, std::map<std::string, MonitorElement*> & );
      std::map<DTStationIndex,std::set<RPCDetId> > rollstoreDT;
      std::map<CSCStationIndex,std::set<RPCDetId> > rollstoreCSC;
   
      MonitorElement * statistics;
      
      //Residuals
      MonitorElement * hGlobalResClu1La[6];
      MonitorElement * hGlobalResClu2La[6];
      MonitorElement * hGlobalResClu3La[6];

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
      //   std::ofstream ofrej;
      bool incldt;
      bool incldtMB4;
      bool inclcsc;
      bool debug;
      //   bool paper;
      bool inves;
      double rangestrips;
      double rangestripsRB4;
      double MinCosAng;
      double MaxD;
      double MaxDrb4;
      int dupli;
  
      edm::EDGetTokenT<CSCSegmentCollection> cscSegments;
      edm::EDGetTokenT<DTRecSegment4DCollection> dt4DSegments;
      edm::EDGetTokenT<RPCRecHitCollection>    RPCRecHitLabel_;
      
       std::string folderPath;
      std::string rollseff;
      
      std::map<int, std::map<std::string, MonitorElement*> >  meCollection;
      
      bool EffSaveRootFile;
      std::string EffRootFileName;
};

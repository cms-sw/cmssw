#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerStackedDigi.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

namespace Phase2Tracker
{

  class Phase2TrackerDigiToRaw
  {
    public:
      Phase2TrackerDigiToRaw() {}
      Phase2TrackerDigiToRaw(const Phase2TrackerCabling *, const TrackerGeometry* tGeom, const TrackerTopology* tTopo, std::map< int, std::pair<int,int> > stackMap, edm::Handle< edmNew::DetSetVector< Phase2TrackerCluster1D > >, int);
      ~Phase2TrackerDigiToRaw() {}
      // loop on FEDs to create buffers
      void buildFEDBuffers(std::auto_ptr<FEDRawDataCollection>&);
      // builds a single FED buffer
      std::vector<uint64_t> makeBuffer(std::vector< edmNew::DetSet< Phase2TrackerCluster1D > >);
      // write FE Header to buffer
      void writeFeHeaderSparsified(std::vector<uint64_t>&,uint64_t&,int,int,int);
      // determine if a P or S cluster should be written
      void writeCluster(std::vector<uint64_t>&,uint64_t&, stackedDigi);
      // write S cluster to buffer
      void writeSCluster(std::vector<uint64_t>&,uint64_t&, stackedDigi);
      // write P cluster to buffer
      void writePCluster(std::vector<uint64_t>&,uint64_t&, stackedDigi);
    private:
      // data you get from outside
      const Phase2TrackerCabling * cabling_; 
      const TrackerTopology* tTopo_;
      const TrackerGeometry* tGeom_;
      std::map< int, std::pair<int,int> > stackMap_;
      edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > digishandle_;
      int mode_;
      // headers 
      FEDDAQHeader FedDaqHeader_;
      FEDDAQTrailer FedDaqTrailer_; 
      Phase2TrackerFEDHeader FedHeader_;
  };
}

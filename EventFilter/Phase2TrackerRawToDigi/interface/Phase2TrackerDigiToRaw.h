#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

namespace Phase2Tracker
{
  class Phase2TrackerDigiToRaw
  {
    public:
      Phase2TrackerDigiToRaw() {}
      Phase2TrackerDigiToRaw(const Phase2TrackerCabling *, const TrackerTopology *, edm::Handle< edmNew::DetSetVector<SiPixelCluster> >, int);
      ~Phase2TrackerDigiToRaw() {}
      void buildFEDBuffers(std::auto_ptr<FEDRawDataCollection>&);
      std::vector<uint64_t> makeBuffer(std::vector<edmNew::DetSet<SiPixelCluster>>);
      void writeFeHeaderSparsified(std::vector<uint64_t>&,uint64_t&,int,int,int);
      void writeSCluster(std::vector<uint64_t>&,uint64_t&,const SiPixelCluster*);
      std::pair<int,int> calcChipId(const SiPixelCluster*);
    private:
      // data you get from outside
      const Phase2TrackerCabling * cabling_; 
      const TrackerTopology * topo_; 
      edm::Handle< edmNew::DetSetVector<SiPixelCluster> > digishandle_;
      int mode_;
      // headers to be created at init 
      FEDDAQHeader FedDaqHeader_;
      FEDDAQTrailer FedDaqTrailer_; 
      Phase2TrackerFEDHeader FedHeader_;
  };
}

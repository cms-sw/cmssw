#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerDigiToRaw.h"
#include "CondFormats/DataRecord/interface/Phase2TrackerCablingRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/DetSetVector.h"

namespace Phase2Tracker
{
  Phase2TrackerDigiToRaw::Phase2TrackerDigiToRaw(const Phase2TrackerCabling * cabling, const TrackerTopology * topo, edm::Handle< edmNew::DetSetVector<SiPixelCluster> > digis_handle, int mode):
    cabling_(cabling),
    topo_(topo),
    digishandle_(digis_handle),
    mode_(mode)
  {
  }

  void Phase2TrackerDigiToRaw::buildFEDBuffers()
  {
    // vector to store digis for a given fedid
    std::vector<edmNew::DetSet<SiPixelCluster>> digis_t;
    // iterate on all possible channels 
    std::vector<Phase2TrackerModule> conns = cabling_->connections();
    std::vector<Phase2TrackerModule>::iterator iconn;
    unsigned int fedid_current = 0;
    for(iconn = conns.begin(); iconn != conns.end(); iconn++) 
    {
      std::pair<unsigned int, unsigned int> fedch = iconn->getCh();
      int detid = iconn->getDetid();
      // std::vector<edm::DetSet<SiPixelCluster>>::const_iterator digis = digishandle_->find(detid);
      edmNew::DetSetVector<SiPixelCluster>::const_iterator  digis = digishandle_->find(detid);
      if(fedch.first == fedid_current) 
      {
        // adding detset to the current fed
        digis_t.push_back(*digis);
      }
      else
      {
        // store create buffer for this fed
        // empty digis_t
        digis_t.empty();
        fedid_current = fedch.first;
      }
    }
  }
}

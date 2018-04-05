#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHit.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMCSCSegment.h"
#include "DataFormats/GEMRecHit/interface/GEMCSCSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegment.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0Segment.h"
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace DataFormats_GEMRecHit {
  struct dictionary {
    std::pair<unsigned int, unsigned int> dummygem1;
    std::pair<unsigned long, unsigned long> dummygem2;
    std::map<GEMDetId, std::pair<unsigned int, unsigned int> > dummygemdetid1;
    std::map<GEMDetId, std::pair<unsigned long, unsigned long> > dummygemdetid2;
    std::map<ME0DetId, std::pair<unsigned int, unsigned int> > dummyme0detid1;
    std::map<ME0DetId, std::pair<unsigned long, unsigned long> > dummyme0detid2;

    GEMRecHit rrh;
    std::vector<GEMRecHit> vrh;
    GEMRecHitCollection c;
    edm::Wrapper<GEMRecHitCollection> w;

    ME0RecHit mrh;
    std::vector<ME0RecHit> vmrh;
    ME0RecHitCollection mc;
    edm::Wrapper<ME0RecHitCollection> mw;

    GEMCSCSegment gs;
    GEMCSCSegmentCollection gseg;
    edm::Wrapper<GEMCSCSegmentCollection> gdwc1;
    GEMCSCSegmentRef gref;

    ME0Segment ms;
    ME0SegmentCollection seg;
    edm::Wrapper<ME0SegmentCollection> dwc1;
    ME0SegmentRef ref;

    GEMSegment gems;
    GEMSegmentCollection gemseg;
    edm::Wrapper<GEMSegmentCollection> gemdwc1;
    GEMCSCSegmentRef gemref;
  };
}


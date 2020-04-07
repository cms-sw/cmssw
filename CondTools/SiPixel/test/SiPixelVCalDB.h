#ifndef CalibTracker_SiPixelVCalDB_SiPixelVCalDB_h
#define CalibTracker_SiPixelVCalDB_SiPixelVCalDB_h
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
//#include "CondTools/SiPixel/test/SiPixelVCalPixelId.h"

class SiPixelVCalDB : public edm::EDAnalyzer {
public:

  explicit SiPixelVCalDB(const edm::ParameterSet& conf);
  explicit SiPixelVCalDB();
  virtual ~SiPixelVCalDB();
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  // "PixelId"
  // BPix: 1000*(subdetId=1) + 100*(layer=1,2,3,4)
  // FPix: 1000*(subdetId=2) + 100*(side=1,2) + 10*(disk=1,2,3) + 1*(ring=1,2)
  enum PixelId {
    L1=1100, L2=1200, L3=1300, L4=1400, // BPix
    Rm1l=2111, Rm1u=2112, Rm2l=2121, Rm2u=2122, Rm3l=2131, Rm3u=2132, // FPix minus
    Rp1l=2211, Rp1u=2212, Rp2l=2221, Rp2u=2222, Rp3l=2231, Rp3u=2232, // FPix plus
  };
  
  static const PixelId calculateBPixID(const unsigned int layer){
    // BPix: 1000*(subdetId=1) + 100*(layer=1,2,3,4)
    PixelId bpixLayer = static_cast<PixelId>(1000+100*layer);
    return bpixLayer;
  }
  
  static const PixelId calculateFPixID(const unsigned int side, const unsigned int disk, const unsigned int ring){
    // FPix: 1000*(subdetId=2) + 100*(side=1,2) + 10*(disk=1,2,3) + 1*(ring=1,2)
    PixelId fpixRing = static_cast<PixelId>(2000+100*side+10*disk+ring);
    return fpixRing;
  }
  
  static const int getPixelSubDetector(const unsigned int pixid){ //SiPixelVCalDB::PixelId
    // subdetId: BPix=1, FPix=2
    return (pixid/1000)%10;
  }
  
  static const PixelId detIdToPixelId(const unsigned int detid, const TrackerTopology* trackTopo, const bool phase1){
    DetId detId = DetId(detid);
    unsigned int subid = detId.subdetId();
    unsigned int pixid = 0;
    if (subid==1) { // BPix static_cast<int>(PixelSubdetector::PixelEndcap)
      PixelBarrelName bpix(detId,trackTopo,phase1);
      int layer = bpix.layerName();
      pixid = calculateBPixID(layer);
    }
    else if (subid==2) { // FPix static_cast<int>(PixelSubdetector::PixelBarrel)
      PixelEndcapName fpix(detId,trackTopo,phase1);
      int side = trackTopo->pxfSide(detId); // 1 (-z), 2 for (+z)
      int disk = fpix.diskName(); //trackTopo->pxfDisk(detId); // 1, 2, 3
      int ring = fpix.ringName(); // 1 (lower), 2 (upper)
      pixid = calculateFPixID(side,disk,ring);
    }
    PixelId pixID = static_cast<PixelId>(pixid);
    return pixID;
  }


private:

  std::string recordName_;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BPixParameters_;
  Parameters FPixParameters_;

};

#endif

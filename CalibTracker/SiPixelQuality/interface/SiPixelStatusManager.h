#ifndef SiPixelStatusManager_H
#define SiPixelStatusManager_H

/** \class SiPixelStatusManager
 *  
 *
 *  \author 
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include <string>
#include <map>
#include <utility>
#include <iostream>

//Data format
#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"

class SiPixelStatusManager {

 public:
  SiPixelStatusManager        ();
  SiPixelStatusManager         (const edm::ParameterSet&, edm::ConsumesCollector&&);
  virtual ~SiPixelStatusManager();

  void reset();   
  void readLumi(const edm::LuminosityBlock&);   

  void createPayloads(){
      createFEDerror25();
      createBadComponents();
  }

  const std::map<edm::LuminosityBlockNumber_t,SiPixelDetectorStatus>& getBadComponents(){return siPixelStatusMap_; } 
  const std::map<edm::LuminosityBlockNumber_t,std::map<int, std::vector<int>> >& getFEDerror25Rocs(){return FEDerror25Map_;}

  typedef std::map<edm::LuminosityBlockNumber_t,SiPixelDetectorStatus>::iterator siPixelStatusMap_iterator;
  typedef std::map<edm::LuminosityBlockNumber_t,std::map<int, std::vector<int>> >::iterator FEDerror25Map_iterator;

 private:

  void createFEDerror25();
  void createBadComponents();

  std::map<edm::LuminosityBlockNumber_t, SiPixelDetectorStatus> siPixelStatusMap_;
  std::map<edm::LuminosityBlockNumber_t, std::map<int, std::vector<int>> > FEDerror25Map_;

  std::string outputBase_;
  int aveDigiOcc_;
  int nLumi_;
  std::string moduleName_;
  std::string label_;

  edm::EDGetTokenT<SiPixelDetectorStatus> siPixelStatusToken_;


};

#endif

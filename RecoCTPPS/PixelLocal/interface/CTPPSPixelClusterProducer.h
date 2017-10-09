/**********************************************************************
 *
 * Author: F.Ferro fabrizio.ferro@ge.infn.it - INFN Genova - 2017
 *
 **********************************************************************/
#ifndef RecoCTPPS_PixelLocal_CTPPSPixelClusterProducer
#define RecoCTPPS_PixelLocal_CTPPSPixelClusterProducer

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"
#include "RecoCTPPS/PixelLocal/interface/CTPPSPixelGainCalibrationDBService.h"
#include "RecoCTPPS/PixelLocal/interface/RPixDetClusterizer.h"

#include <vector>
#include <set>

class CTPPSPixelClusterProducer : public edm::stream::EDProducer<>
{
public:
  explicit CTPPSPixelClusterProducer(const edm::ParameterSet& param);
 
  ~CTPPSPixelClusterProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  edm::ParameterSet param_;
  int verbosity_;
 
  edm::InputTag src_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelDigi>> tokenCTPPSPixelDigi_;

  RPixDetClusterizer clusterizer_;
  
  void run(const edm::DetSetVector<CTPPSPixelDigi> &input, edm::DetSetVector<CTPPSPixelCluster> &output, const CTPPSPixelAnalysisMask *mask);

  CTPPSPixelGainCalibrationDBService theGainCalibrationDB; 
};



#endif

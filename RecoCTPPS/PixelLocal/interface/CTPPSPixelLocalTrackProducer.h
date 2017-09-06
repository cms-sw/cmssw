/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*   Enrico Robutti (robutti@ge.infn.it)
*   Fabio Ravera   (fabio.ravera@cern.ch)
*
*/
#ifndef RecoCTPPS_PixelLocal_CTPPSPixelLocalTrackProducer_H
#define RecoCTPPS_PixelLocal_CTPPSPixelLocalTrackProducer_H

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
 
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/DetId/interface/DetId.h"


#include "RecoCTPPS/PixelLocal/interface/RPixDetPatternFinder.h"
#include "RecoCTPPS/PixelLocal/interface/RPixDetTrackFinder.h"

#include <string>
#include <vector>

class CTPPSPixelLocalTrackProducer : public edm::stream::EDProducer<>
{
public:
  explicit CTPPSPixelLocalTrackProducer(const edm::ParameterSet& parameterSet);
 
  ~CTPPSPixelLocalTrackProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  edm::ParameterSet parameterSet_;
  int verbosity_;
  int maxHitPerPlane_;
  int maxHitPerRomanPot_;
  int maxTrackPerRomanPot_;
  int maxTrackPerPattern_;
 
  edm::InputTag inputTag_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelRecHit>> tokenCTPPSPixelRecHit_;
  edm::ESWatcher<VeryForwardRealGeometryRecord> geometryWatcher_;
  uint32_t numberOfPlanesPerPot_;
  std::vector<uint32_t> listOfAllPlanes_;

  RPixDetPatternFinder *patternFinder_;
  RPixDetTrackFinder   *trackFinder_;
  
  void run(const edm::DetSetVector<CTPPSPixelRecHit> &input, edm::DetSetVector<CTPPSPixelLocalTrack> &output);
  
};



#endif

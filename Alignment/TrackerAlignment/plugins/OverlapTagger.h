#include <Riostream.h>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "Utilities/General/interface/ClassName.h"

#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"
//#include <boost/regex.hpp>

class OverlapTagger : public edm::EDProducer {
 public:
  OverlapTagger(const edm::ParameterSet &iConfig);
  ~OverlapTagger();
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup);

 private:
  edm::InputTag src_;
  edm::InputTag srcClust_;
  bool rejectBadMods_;
  std::vector<unsigned int> BadModsList_;


  int layerFromId (const DetId& id) const;
};

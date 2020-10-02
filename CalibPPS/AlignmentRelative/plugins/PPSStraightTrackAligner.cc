/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*  Cristian Baldenegro (crisx.baldenegro@gmail.com)
****************************************************************************/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CalibPPS/AlignmentRelative/interface/StraightTrackAlignment.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

/**
 *\brief An EDAnalyzer that runs StraightTrackAlignment.
 **/
class PPSStraightTrackAligner : public edm::EDAnalyzer
{
  public:
    PPSStraightTrackAligner(const edm::ParameterSet &ps); 
    ~PPSStraightTrackAligner() {}

  private:
    unsigned int verbosity_;

    edm::InputTag tagUVPatternsStrip_;
    edm::InputTag tagDiamondHits_;
    edm::InputTag tagPixelHits_;
    edm::InputTag tagPixelLocalTracks_;

    edm::EDGetTokenT<edm::DetSetVector<TotemRPUVPattern>> tokenUVPatternsStrip_;
    edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondRecHit>> tokenDiamondHits_;
    edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelRecHit>> tokenPixelHits_;
    edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>> tokenPixelLocalTracks_;

    bool worker_initialized_;
    StraightTrackAlignment worker_;

    edm::ESWatcher<VeryForwardRealGeometryRecord> geometryWatcher_;

    virtual void beginJob() override {}

    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;

    virtual void analyze(const edm::Event &e, const edm::EventSetup &es) override;

    virtual void endJob() override;
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

PPSStraightTrackAligner::PPSStraightTrackAligner(const ParameterSet &ps) : 
  verbosity_(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),

  tagUVPatternsStrip_(ps.getParameter<edm::InputTag>("tagUVPatternsStrip")),
  tagDiamondHits_(ps.getParameter<edm::InputTag>("tagDiamondHits")),
  tagPixelHits_(ps.getParameter<edm::InputTag>("tagPixelHits")),
  tagPixelLocalTracks_(ps.getParameter<edm::InputTag>("tagPixelLocalTracks")),

  tokenUVPatternsStrip_(consumes<DetSetVector<TotemRPUVPattern>>(tagUVPatternsStrip_)),
  tokenDiamondHits_(consumes<DetSetVector<CTPPSDiamondRecHit>>(tagDiamondHits_)),
  tokenPixelHits_(consumes<DetSetVector<CTPPSPixelRecHit>>(tagPixelHits_)),
  tokenPixelLocalTracks_(consumes<DetSetVector<CTPPSPixelLocalTrack>>(tagPixelLocalTracks_)),

  worker_initialized_(false),
  worker_(ps)
{
  if (!tagPixelHits_.label().empty() && !tagPixelLocalTracks_.label().empty())
    LogWarning("PPS") << "Both tagPixelHits and tagPixelLocalTracks are not empty - most likely this not what you want.";
}

//----------------------------------------------------------------------------------------------------

void PPSStraightTrackAligner::beginRun(edm::Run const&, edm::EventSetup const& es)
{
}

//----------------------------------------------------------------------------------------------------

void PPSStraightTrackAligner::analyze(const edm::Event &event, const edm::EventSetup &es)
{
  // check if geometry hasn't changed
  if (geometryWatcher_.check(es))
  {
    if (worker_initialized_)
      throw cms::Exception("PPS") <<
        "PPSStraightTrackAligner can't cope with changing geometry - change in event " << event.id() << endl;
  }

  // check if worker already initialised
  if (!worker_initialized_)
  {
    worker_.begin(es);
    worker_initialized_ = true;
  }

  // get input
  DetSetVector<TotemRPUVPattern> defaultStripUVPatterns;
  const DetSetVector<TotemRPUVPattern> *pStripUVPatterns = &defaultStripUVPatterns;
  Handle<DetSetVector<TotemRPUVPattern>> inputStripUVPatterns;
  if (!tagUVPatternsStrip_.label().empty())
  {
    event.getByToken(tokenUVPatternsStrip_, inputStripUVPatterns);
    pStripUVPatterns = &(*inputStripUVPatterns);
  }

  DetSetVector<CTPPSDiamondRecHit> defaultDiamondHits;
  const DetSetVector<CTPPSDiamondRecHit> *pDiamondHits = &defaultDiamondHits;
  Handle<DetSetVector<CTPPSDiamondRecHit>> inputDiamondHits;
  if (!tagDiamondHits_.label().empty())
  {
    event.getByToken(tokenDiamondHits_, inputDiamondHits);
    pDiamondHits = &(*inputDiamondHits);
  }

  DetSetVector<CTPPSPixelRecHit> defaultPixelHits;
  const DetSetVector<CTPPSPixelRecHit> *pPixelHits = &defaultPixelHits;
  Handle<DetSetVector<CTPPSPixelRecHit>> inputPixelHits;
  if (!tagPixelHits_.label().empty())
  {
    event.getByToken(tokenPixelHits_, inputPixelHits);
    pPixelHits = &(*inputPixelHits);
  }

  DetSetVector<CTPPSPixelLocalTrack> defaultPixelLocalTracks;
  const DetSetVector<CTPPSPixelLocalTrack> *pPixelLocalTracks = &defaultPixelLocalTracks;
  Handle<DetSetVector<CTPPSPixelLocalTrack>> inputPixelLocalTracks;
  if (!tagPixelLocalTracks_.label().empty())
  {
    event.getByToken(tokenPixelLocalTracks_, inputPixelLocalTracks);
    pPixelLocalTracks = &(*inputPixelLocalTracks);
  }

  // feed worker
  worker_.processEvent(event.id(), *pStripUVPatterns, *pDiamondHits, *pPixelHits, *pPixelLocalTracks);
}

//----------------------------------------------------------------------------------------------------

void PPSStraightTrackAligner::endJob()
{
  if (worker_initialized_)
    worker_.finish();
  else
    throw cms::Exception("PPS") <<
      "worker not initialized." << endl;
}

DEFINE_FWK_MODULE(PPSStraightTrackAligner);

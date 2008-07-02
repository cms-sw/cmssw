#include "EnergyLossProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/EnergyLossPlain.h"

#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"

/*****************************************************************************/
EnergyLossProducer::EnergyLossProducer(const edm::ParameterSet& ps)
{
  trackProducer          = ps.getParameter<string>("trackProducer");
  pixelToStripMultiplier = ps.getParameter<double>("pixelToStripMultiplier");
  pixelToStripExponent   = ps.getParameter<double>("pixelToStripExponent");

  produces<reco::TrackDeDxEstimateCollection>();
}

/*****************************************************************************/
EnergyLossProducer::~EnergyLossProducer()
{
}

/*****************************************************************************/
void EnergyLossProducer::beginJob(const edm::EventSetup& es)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker = tracker.product();
}

/*****************************************************************************/
void EnergyLossProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<reco::TrackCollection> trackHandle;
  ev.getByLabel(trackProducer,       trackHandle);

  auto_ptr<reco::TrackDeDxEstimateCollection>
    output(new reco::TrackDeDxEstimateCollection(reco::TrackRefProd(trackHandle)));

  LogTrace("MinBiasTracking")
    << "[EnergyLossProducer]";

  // Get trajectory collection
  edm::Handle<vector<Trajectory> > trajeHandle;
  ev.getByLabel(trackProducer,     trajeHandle);
  const vector<Trajectory> & trajeCollection =
                                 *(trajeHandle.product());

  // Plain estimator
  EnergyLossPlain theEloss(theTracker, pixelToStripMultiplier,
                                       pixelToStripExponent);

  // Take all trajectories
  int j = 0;
  for(vector<Trajectory>::const_iterator traje = trajeCollection.begin();
                                         traje!= trajeCollection.end();
                                         traje++, j++)
  {
    vector<pair<int,double> >    arithmeticMean, truncatedMean;
    theEloss.estimate(&(*traje), arithmeticMean, truncatedMean);

    // use all hits, give truncated mean
    output->setValue(j, truncatedMean[2].second);

// !!!!
/*
    cerr << " dedx " << truncatedMean[0].second
              << " " << truncatedMean[1].second
              << " " << truncatedMean[2].second << endl;
*/
  }

  // Put back result to event
  ev.put(output);
}

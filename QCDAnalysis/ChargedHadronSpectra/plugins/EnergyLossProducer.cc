#include "EnergyLossProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "QCDAnalysis/ChargedHadronSpectra/interface/EnergyLossPlain.h"

#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"

#include "TROOT.h"
#include "TFile.h"
#include "TH2F.h"

/*****************************************************************************/
EnergyLossProducer::EnergyLossProducer(const edm::ParameterSet& ps)
{
  trackProducer          = ps.getParameter<string>("trackProducer");
  pixelToStripMultiplier = ps.getParameter<double>("pixelToStripMultiplier");
  pixelToStripExponent   = ps.getParameter<double>("pixelToStripExponent");

  produces<reco::TrackDeDxEstimateCollection>("energyLossPixHits");
  produces<reco::TrackDeDxEstimateCollection>("energyLossStrHits");
  produces<reco::TrackDeDxEstimateCollection>("energyLossAllHits");

  resultFile = new TFile("energyLoss.root","recreate");
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
  
  std::vector<double> ldeBins;
  static float ldeMin   = log(1);
  static float ldeMax   = log(100);
  static float ldeWidth = (ldeMax - ldeMin)/250;
  for(double lde = ldeMin; lde < ldeMax + ldeWidth/2; lde += ldeWidth)
    ldeBins.push_back(lde);

  hnor = new TH2F("hnor","hnor", ldeBins.size()-1, &ldeBins[0],
                                 ldeBins.size()-1, &ldeBins[0]);
}

/*****************************************************************************/
void EnergyLossProducer::endJob()
{
  resultFile->cd();

  hnor->Write();

  resultFile->Close();
}

/*****************************************************************************/
void EnergyLossProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<reco::TrackCollection> trackHandle;
  ev.getByLabel(trackProducer,       trackHandle);

  auto_ptr<reco::TrackDeDxEstimateCollection> outputPix
      (new reco::TrackDeDxEstimateCollection(reco::TrackRefProd(trackHandle)));
  auto_ptr<reco::TrackDeDxEstimateCollection> outputStr
      (new reco::TrackDeDxEstimateCollection(reco::TrackRefProd(trackHandle)));
  auto_ptr<reco::TrackDeDxEstimateCollection> outputAll
      (new reco::TrackDeDxEstimateCollection(reco::TrackRefProd(trackHandle)));

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
    // Estimate (nhits,dE/dx)
    vector<pair<int,double> >    arithmeticMean, weightedMean;
    theEloss.estimate(&(*traje), arithmeticMean, weightedMean);

    // Set values
    outputPix->setValue(j, Measurement1D(weightedMean[0].second,
                                         weightedMean[0].first));
    outputStr->setValue(j, Measurement1D(weightedMean[1].second,
                                         weightedMean[1].first));
    outputAll->setValue(j, Measurement1D(weightedMean[2].second,
                                         weightedMean[2].first));

    // Prepare conversion matrix
    if(weightedMean[0].first >= 3 &&
       weightedMean[1].first >= 3)
      hnor->Fill(log(weightedMean[0].second),
                 log(weightedMean[1].second));
  }

  // Put back result to event
  ev.put(outputPix, "energyLossPixHits");
  ev.put(outputStr, "energyLossStrHits");
  ev.put(outputAll, "energyLossAllHits");
}

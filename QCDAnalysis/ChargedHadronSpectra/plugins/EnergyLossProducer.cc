#include "EnergyLossProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "QCDAnalysis/ChargedHadronSpectra/interface/EnergyLossPlain.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

//#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
//#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"

#include "TROOT.h"
#include "TFile.h"
#include "TH2F.h"

/*****************************************************************************/
EnergyLossProducer::EnergyLossProducer(const edm::ParameterSet& ps)
{
  trackProducer          = ps.getParameter<std::string>("trackProducer");
//  pixelToStripMultiplier = ps.getParameter<double>("pixelToStripMultiplier");
//  pixelToStripExponent   = ps.getParameter<double>("pixelToStripExponent");

  produces<reco::DeDxDataValueMap>("energyLossPixHits");
  produces<reco::DeDxDataValueMap>("energyLossStrHits");
  produces<reco::DeDxDataValueMap>("energyLossAllHits");

  resultFile = new TFile("energyLoss.root","recreate");
}

/*****************************************************************************/
EnergyLossProducer::~EnergyLossProducer()
{
}

/*****************************************************************************/
void EnergyLossProducer::beginRun(const edm::Run & run, const edm::EventSetup& es)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker = tracker.product();
  
  std::vector<double> ldeBins;
  float ldeMin   = log(1);
  float ldeMax   = log(100);
  float ldeWidth = (ldeMax - ldeMin)/250;
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

  std::auto_ptr<reco::DeDxDataValueMap> outputPix (new reco::DeDxDataValueMap);
  std::auto_ptr<reco::DeDxDataValueMap> outputStr (new reco::DeDxDataValueMap);
  std::auto_ptr<reco::DeDxDataValueMap> outputAll (new reco::DeDxDataValueMap);

  reco::DeDxDataValueMap::Filler fillerPix(*outputPix);
  reco::DeDxDataValueMap::Filler fillerStr(*outputStr);
  reco::DeDxDataValueMap::Filler fillerAll(*outputAll);

  LogTrace("MinBiasTracking")
    << "[EnergyLossProducer]";

  // Get trajectory collection
  edm::Handle<std::vector<Trajectory> > trajeHandle;
  ev.getByLabel(trackProducer,     trajeHandle);
  const std::vector<Trajectory> & trajeCollection =
                                 *(trajeHandle.product());

  // Plain estimator
  EnergyLossPlain theEloss(theTracker, pixelToStripMultiplier,
                                       pixelToStripExponent);

  std::vector<reco::DeDxData> estimatePix;
  std::vector<reco::DeDxData> estimateStr;
  std::vector<reco::DeDxData> estimateAll;

  // Take all trajectories
  int j = 0;
  for(std::vector<Trajectory>::const_iterator traje = trajeCollection.begin();
                                         traje!= trajeCollection.end();
                                         traje++, j++)
  {
    // Estimate (nhits,dE/dx)
    std::vector<std::pair<int,double> >    arithmeticMean, weightedMean;
    theEloss.estimate(&(*traje), arithmeticMean, weightedMean);

    // Set values
    estimatePix.push_back(reco::DeDxData(weightedMean[0].second, 0,
                                         weightedMean[0].first));
    estimateStr.push_back(reco::DeDxData(weightedMean[1].second, 0,
                                         weightedMean[1].first));
    estimateAll.push_back(reco::DeDxData(weightedMean[2].second, 0,
                                         weightedMean[2].first));

    // Prepare conversion matrix
    if(weightedMean[0].first >= 3 &&
       weightedMean[1].first >= 3)
      hnor->Fill(log(weightedMean[0].second),
                 log(weightedMean[1].second));
  }

  fillerPix.insert(trackHandle, estimatePix.begin(), estimatePix.end());
  fillerStr.insert(trackHandle, estimateStr.begin(), estimateStr.end());
  fillerAll.insert(trackHandle, estimateAll.begin(), estimateAll.end());

  fillerPix.fill();
  fillerStr.fill();
  fillerAll.fill();

  // Put back result to event
  ev.put(outputPix, "energyLossPixHits");
  ev.put(outputStr, "energyLossStrHits");
  ev.put(outputAll, "energyLossAllHits");
}

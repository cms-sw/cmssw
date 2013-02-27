#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/EnergyLossPlain.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"

#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"

#include <fstream>
using namespace std;

/*****************************************************************************/
class EnergyLossEstimator : public edm::EDAnalyzer
{
 public:
   explicit EnergyLossEstimator(const edm::ParameterSet& pset);
   ~EnergyLossEstimator();
   virtual void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
   virtual void analyze(const edm::Event& ev, const edm::EventSetup& es) override;
   virtual void endJob();

 private:
   double pixelToStripMultiplier, pixelToStripExponent;

   const TrackerGeometry * theTracker;
   TFile * resultFile;
   TNtuple * energyLoss;

   const TrackAssociatorByHits * theAssociatorByHits;
};

/*****************************************************************************/
EnergyLossEstimator::EnergyLossEstimator(const edm::ParameterSet& pset)
{
  pixelToStripMultiplier = pset.getParameter<double>("pixelToStripMultiplier");
  pixelToStripExponent   = pset.getParameter<double>("pixelToStripExponent");

  std::string resultFileLabel = pset.getParameter<std::string>("resultFile");

  resultFile = new TFile(resultFileLabel.c_str(),"RECREATE");
  resultFile->cd();
  energyLoss = new TNtuple("energyLoss","energyLoss",
    "p:npix:tpix:nstr:tstr:nall:tall:psim:id");
}

/*****************************************************************************/
EnergyLossEstimator::~EnergyLossEstimator()
{
}

/*****************************************************************************/
void EnergyLossEstimator::beginRun(const edm::EventSetup& es)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker = tracker.product();

  // Get associator
  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  es.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",
                                    theHitsAssociator);
  theAssociatorByHits =
   (const TrackAssociatorByHits*)theHitsAssociator.product();
}

/*****************************************************************************/
void EnergyLossEstimator::endJob()
{
  resultFile->cd();
  energyLoss->Write();
  resultFile->Close();
}

/*****************************************************************************/
void EnergyLossEstimator::analyze
  (const edm::Event& ev, const edm::EventSetup& es)
{
  // Get trajectories
  edm::Handle<std::vector<Trajectory> >  trajCollection;
  ev.getByLabel("globalPrimTracks", trajCollection);
  const std::vector<Trajectory>* trajs = trajCollection.product();
  std::cerr << "[EnergyLossEstimator] trajectories = " << trajs->size() << std::endl;

  // Get simulated
  edm::Handle<TrackingParticleCollection> simCollection;
  ev.getByType(simCollection);

  // Get reconstructed
  edm::Handle<edm::View<reco::Track> >  recCollection;
  ev.getByLabel("globalPrimTracks", recCollection);

  // Associatiors
   reco::RecoToSimCollection recoToSim =
    theAssociatorByHits->associateRecoToSim(recCollection, simCollection,&ev);

  // Plain estimator
  EnergyLossPlain theEloss(theTracker, pixelToStripMultiplier,
                                       pixelToStripExponent);

  // Take all trajectories
  int i = 0;
  for(std::vector<Trajectory>::const_iterator traj = trajs->begin();
                                         traj!= trajs->end(); traj++)
  {
    std::vector<pair<int,double> > arithmeticMean, truncatedMean;

    double p = traj->firstMeasurement().updatedState().globalMomentum().mag();

    theEloss.estimate(&(*traj), arithmeticMean, truncatedMean);

    std::vector<float> result;

    result.push_back(p);                       // p
    result.push_back(truncatedMean[0].first);  // npix
    result.push_back(truncatedMean[0].second); // tpix
    result.push_back(truncatedMean[1].first);  // nstr
    result.push_back(truncatedMean[1].second); // tstr
    result.push_back(truncatedMean[2].first);  // nall
    result.push_back(truncatedMean[2].second); // tall

    // Match
    TrackingParticleRef matchedSimTrack;
    int nSim = 0;

    edm::RefToBase<reco::Track> recTrack(recCollection, i);
    std::vector<pair<TrackingParticleRef, double> > simTracks = recoToSim[recTrack];
    for(std::vector<pair<TrackingParticleRef, double> >::const_iterator
            it = simTracks.begin(); it != simTracks.end(); ++it)
    {
      TrackingParticleRef simTrack = it->first;
      float fraction = it->second;

      // If more than half is shared
      if(fraction > 0.5)
      { matchedSimTrack = simTrack; nSim++; }
    }


    if(nSim > 0)
    {
      result.push_back(matchedSimTrack->momentum().R()); // psim
      result.push_back(matchedSimTrack->pdgId()); // id
    }
    else
    {     
      result.push_back(p);   // psim
      result.push_back(211); // id
    }

    energyLoss->Fill(&result[0]);

    i++;
  }
}

DEFINE_FWK_MODULE(EnergyLossEstimator);

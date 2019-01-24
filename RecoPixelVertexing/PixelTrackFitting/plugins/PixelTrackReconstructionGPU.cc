#include <vector>

#include <cuda_runtime.h>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackBuilder.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerWrapper.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h" // for helix_fit
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "PixelTrackReconstructionGPU.h"

using namespace pixeltrackfitting;
using edm::ParameterSet;

PixelTrackReconstructionGPU::PixelTrackReconstructionGPU(const ParameterSet& cfg,
	   edm::ConsumesCollector && iC)
  : theHitSetsToken(iC.consumes<RegionsSeedingHitSets>(cfg.getParameter<edm::InputTag>("SeedingHitSets"))),
    theFitterToken(iC.consumes<PixelFitter>(cfg.getParameter<edm::InputTag>("Fitter"))),
    theCleanerName(cfg.getParameter<std::string>("Cleaner"))
{
  edm::InputTag filterTag = cfg.getParameter<edm::InputTag>("Filter");
  if(!filterTag.label().empty()) {
    theFilterToken = iC.consumes<PixelTrackFilter>(filterTag);
  }
}

PixelTrackReconstructionGPU::~PixelTrackReconstructionGPU()
{
}

void PixelTrackReconstructionGPU::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<edm::InputTag>("SeedingHitSets", edm::InputTag("pixelTracksHitTriplets"));
  desc.add<edm::InputTag>("Fitter", edm::InputTag("pixelFitterByHelixProjections"));
  desc.add<edm::InputTag>("Filter", edm::InputTag("pixelTrackFilterByKinematics"));
  desc.add<std::string>("Cleaner", "pixelTrackCleanerBySharedHits");
}

void PixelTrackReconstructionGPU::run(TracksWithTTRHs& tracks,
    edm::Event& ev, const edm::EventSetup& es)
{
  edm::ESHandle<MagneticField> fieldESH;
  es.get<IdealMagneticFieldRecord>().get(fieldESH);

 edm::Handle<RegionsSeedingHitSets> hhitSets;
  ev.getByToken(theHitSetsToken, hhitSets);
  const auto& hitSets = *hhitSets;

  const PixelTrackFilter *filter = nullptr;
  if(!theFilterToken.isUninitialized()) {
    edm::Handle<PixelTrackFilter> hfilter;
    ev.getByToken(theFilterToken, hfilter);
    filter = hfilter.product();
  }

  float bField = 1 / PixelRecoUtilities::fieldInInvGev(es);

  std::vector<float> hits_and_covariances;
  float * hits_and_covariancesGPU = nullptr;
  Rfit::helix_fit * helix_fit_results = nullptr;
  Rfit::helix_fit * helix_fit_resultsGPU = nullptr;

  const int points_in_seed = 4;
  // We use 3 floats for GlobalPosition and 6 floats for GlobalError (that's what is used by the Riemann fit).
  // Assume a safe maximum of 3K seeds: it will dynamically grow, if needed.
  int total_seeds = 0;
  hits_and_covariances.reserve(sizeof(float)*3000*(points_in_seed*12));
  for (auto const & regionHitSets : hitSets) {
    const TrackingRegion& region = regionHitSets.region();
    for (auto const & tuplet : regionHitSets) {
      for (unsigned int iHit = 0; iHit < tuplet.size(); ++iHit) {
        auto const& recHit = tuplet[iHit];
        auto point = GlobalPoint(recHit->globalPosition().basicVector() - region.origin().basicVector());
        auto errors = recHit->globalPositionError();
        hits_and_covariances.push_back(point.x());
        hits_and_covariances.push_back(point.y());
        hits_and_covariances.push_back(point.z());
        hits_and_covariances.push_back(errors.cxx());
        hits_and_covariances.push_back(errors.cyx());
        hits_and_covariances.push_back(errors.cyy());
        hits_and_covariances.push_back(errors.czx());
        hits_and_covariances.push_back(errors.czy());
        hits_and_covariances.push_back(errors.czz());
      }
      total_seeds++;
    }
  }

  // We pretend to have one fit for every seed
  cudaCheck(cudaMallocHost(&helix_fit_results, sizeof(Rfit::helix_fit)*total_seeds));
  cudaCheck(cudaMalloc(&hits_and_covariancesGPU, sizeof(float)*hits_and_covariances.size()));
  cudaCheck(cudaMalloc(&helix_fit_resultsGPU, sizeof(Rfit::helix_fit)*total_seeds));
  cudaCheck(cudaMemset(helix_fit_resultsGPU, 0x00, sizeof(Rfit::helix_fit)*total_seeds ));
  // CUDA MALLOC OF HITS AND COV AND HELIX_FIT RESULTS

  // CUDA MEMCOPY HOST2DEVICE OF HITS AND COVS AND HELIX_FIT RESULTS
  cudaCheck(cudaMemcpy(hits_and_covariancesGPU, hits_and_covariances.data(),
      sizeof(float)*hits_and_covariances.size(), cudaMemcpyDefault));

  // LAUNCH THE KERNEL FIT
  launchKernelFit(hits_and_covariancesGPU, 12*4*total_seeds, 4,
      bField, helix_fit_resultsGPU);
  // CUDA MEMCOPY DEVICE2HOST OF HELIX_FIT
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaMemcpy(helix_fit_results, helix_fit_resultsGPU,
      sizeof(Rfit::helix_fit)*total_seeds, cudaMemcpyDefault));

  cudaCheck(cudaFree(hits_and_covariancesGPU));
  cudaCheck(cudaFree(helix_fit_resultsGPU));
  // Loop on results, create tracks, filter them and pass them down the chain.
  // In order to avoid huge mess with indexing and remembering who did what,
  // simply iterate again over the main containers in the same order, since we
  // are guaranteed that the results have been packed following the very same
  // order. If not, we are doomed.
  PixelTrackBuilder builder;
  int counter = 0;
  std::vector<const TrackingRecHit *> hits;
  hits.reserve(4);
  for (auto const & regionHitSets : hitSets) {
    const TrackingRegion& region = regionHitSets.region();
    for(const SeedingHitSet& tuplet: regionHitSets) {
      auto nHits = tuplet.size(); hits.resize(nHits);

      for (unsigned int iHit = 0; iHit < nHits; ++iHit)
      {
          hits[iHit] = tuplet[iHit];
      }
      auto const &fittedTrack = helix_fit_results[counter];
      counter++;
      int iCharge       = fittedTrack.q;
      float valPhi      = fittedTrack.par(0);
      float valTip      = fittedTrack.par(1);
      float valPt       = fittedTrack.par(2);
      float valCotTheta = fittedTrack.par(3);
      float valZip      = fittedTrack.par(4);

      //  PixelTrackErrorParam param(valEta, valPt);
      float errValPhi = std::sqrt(fittedTrack.cov(0, 0));
      float errValTip = std::sqrt(fittedTrack.cov(1, 1));
      float errValPt = std::sqrt(fittedTrack.cov(2, 2));
      float errValCotTheta = std::sqrt(fittedTrack.cov(3, 3));
      float errValZip = std::sqrt(fittedTrack.cov(4, 4));

      float chi2 = fittedTrack.chi2_line;

      Measurement1D phi(valPhi, errValPhi);
      Measurement1D tip(valTip, errValTip);

      Measurement1D pt(valPt, errValPt);
      Measurement1D cotTheta(valCotTheta, errValCotTheta);
      Measurement1D zip(valZip, errValZip);

      std::unique_ptr<reco::Track> track(
          builder.build(pt, phi, cotTheta, tip, zip, chi2, iCharge, hits,
            fieldESH.product(), region.origin()));
      if (!track) continue;

      if (filter) {
        if (!(*filter)(track.get(), hits)) {
          continue;
        }
      }
      // add tracks
      tracks.emplace_back(track.release(), tuplet);
    }
  }

  cudaCheck(cudaFreeHost(helix_fit_results));

  // skip overlapped tracks
  if(!theCleanerName.empty()) {
    edm::ESHandle<PixelTrackCleaner> hcleaner;
    es.get<PixelTrackCleaner::Record>().get(theCleanerName, hcleaner);
    const auto& cleaner = *hcleaner;
    if(cleaner.fast())
      cleaner.cleanTracks(tracks);
    else
      tracks = PixelTrackCleanerWrapper(&cleaner).clean(tracks);
  }
}

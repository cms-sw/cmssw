/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
* 	Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "RecoCTPPS/TotemRPLocal/interface/TotemRPLocalTrackFitterAlgorithm.h"

//----------------------------------------------------------------------------------------------------

/**
 *\brief Fits tracks trough a single RP.
 **/
class TotemRPLocalTrackFitter : public edm::stream::EDProducer<> {
public:
  explicit TotemRPLocalTrackFitter(const edm::ParameterSet &conf);

  ~TotemRPLocalTrackFitter() override {}

  void produce(edm::Event &e, const edm::EventSetup &c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  int verbosity_;

  /// Selection of the pattern-recognition module.
  edm::InputTag tagUVPattern;

  edm::EDGetTokenT<edm::DetSetVector<TotemRPUVPattern>> patternCollectionToken;

  /// A watcher to detect geometry changes.
  edm::ESWatcher<VeryForwardRealGeometryRecord> geometryWatcher;

  /// The instance of the fitter module
  TotemRPLocalTrackFitterAlgorithm fitter_;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

TotemRPLocalTrackFitter::TotemRPLocalTrackFitter(const edm::ParameterSet &conf)
    : verbosity_(conf.getParameter<int>("verbosity")), fitter_(conf) {
  tagUVPattern = conf.getParameter<edm::InputTag>("tagUVPattern");
  patternCollectionToken = consumes<DetSetVector<TotemRPUVPattern>>(tagUVPattern);

  produces<DetSetVector<TotemRPLocalTrack>>();
}

//----------------------------------------------------------------------------------------------------

void TotemRPLocalTrackFitter::produce(edm::Event &e, const edm::EventSetup &setup) {
  if (verbosity_ > 5)
    LogVerbatim("TotemRPLocalTrackFitter") << ">> TotemRPLocalTrackFitter::produce";

  // get geometry
  edm::ESHandle<CTPPSGeometry> geometry;
  setup.get<VeryForwardRealGeometryRecord>().get(geometry);

  if (geometryWatcher.check(setup))
    fitter_.reset();

  // get input
  edm::Handle<DetSetVector<TotemRPUVPattern>> input;
  e.getByToken(patternCollectionToken, input);

  // run fit for each RP
  DetSetVector<TotemRPLocalTrack> output;

  for (const auto &rpv : *input) {
    CTPPSDetId rpId(rpv.detId());

    // is U-V association unique?
    unsigned int n_U = 0, n_V = 0;
    unsigned int idx_U = 0, idx_V = 0;
    for (unsigned int pi = 0; pi < rpv.size(); pi++) {
      const TotemRPUVPattern &pattern = rpv[pi];

      // here it would make sense to skip non-fittable patterns, but to keep the logic
      // equivalent to version 7_0_4, nothing is skipped
      /*
      if (pattern.fittable() == false)
        continue;
      */

      switch (pattern.projection()) {
        case TotemRPUVPattern::projU:
          n_U++;
          idx_U = pi;
          break;

        case TotemRPUVPattern::projV:
          n_V++;
          idx_V = pi;
          break;

        default:
          break;
      }
    }

    if (n_U != 1 || n_V != 1) {
      if (verbosity_)
        LogVerbatim("TotemRPLocalTrackFitter")
            << ">> TotemRPLocalTrackFitter::produce > Impossible to combine U and V patterns in RP " << rpId
            << " (n_U=" << n_U << ", n_V=" << n_V << ").";

      continue;
    }

    // again, to follow the logic from version 7_0_4, skip the non-fittable patterns here
    if (!rpv[idx_U].fittable() || !rpv[idx_V].fittable())
      continue;

    // combine U and V hits
    DetSetVector<TotemRPRecHit> hits;
    for (auto &ids : rpv[idx_U].hits()) {
      auto &ods = hits.find_or_insert(ids.detId());
      for (auto &h : ids)
        ods.push_back(h);
    }

    for (auto &ids : rpv[idx_V].hits()) {
      auto &ods = hits.find_or_insert(ids.detId());
      for (auto &h : ids)
        ods.push_back(h);
    }

    // run fit
    double z0 = geometry->rpTranslation(rpId).z();

    TotemRPLocalTrack track;
    fitter_.fitTrack(hits, z0, *geometry, track);

    DetSet<TotemRPLocalTrack> &ds = output.find_or_insert(rpId);
    ds.push_back(track);

    if (verbosity_ > 5) {
      unsigned int n_hits = 0;
      for (auto &hds : track.hits())
        n_hits += hds.size();

      LogVerbatim("TotemRPLocalTrackFitter")
          << "    track in RP " << rpId << ": valid = " << track.isValid() << ", hits = " << n_hits;
    }
  }

  // save results
  e.put(make_unique<DetSetVector<TotemRPLocalTrack>>(output));
}

//----------------------------------------------------------------------------------------------------

void TotemRPLocalTrackFitter::fillDescriptions(edm::ConfigurationDescriptions &descr) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("tagUVPattern", edm::InputTag("totemRPUVPatternFinder"))
      ->setComment("input U-V patterns collection to retrieve");
  desc.add<int>("verbosity", 0);

  descr.add("totemRPLocalTrackFitter", desc);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemRPLocalTrackFitter);

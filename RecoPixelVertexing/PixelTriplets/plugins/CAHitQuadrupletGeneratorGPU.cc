//
// Author: Felice Pantaleo, CERN
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CAHitQuadrupletGeneratorGPU.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "CellularAutomaton.h"

#include "CommonTools/Utils/interface/DynArray.h"

#include "FWCore/Utilities/interface/isFinite.h"

#include <functional>

namespace {

  template <typename T> T sqr(T x) { return x * x; }

} // namespace

using namespace std;

constexpr unsigned int CAHitQuadrupletGeneratorGPU::minLayers;

CAHitQuadrupletGeneratorGPU::CAHitQuadrupletGeneratorGPU(
    const edm::ParameterSet &cfg,
    edm::ConsumesCollector &iC)
    : extraHitRPhitolerance(cfg.getParameter<double>(
          "extraHitRPhitolerance")), // extra window in
                                     // ThirdHitPredictionFromCircle range
                                     // (divide by R to get phi)
      maxChi2(cfg.getParameter<edm::ParameterSet>("maxChi2")),
      fitFastCircle(cfg.getParameter<bool>("fitFastCircle")),
      fitFastCircleChi2Cut(cfg.getParameter<bool>("fitFastCircleChi2Cut")),
      useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection")),
      caThetaCut(cfg.getParameter<double>("CAThetaCut")),
      caPhiCut(cfg.getParameter<double>("CAPhiCut")),
      caHardPtCut(cfg.getParameter<double>("CAHardPtCut")) {
  edm::ParameterSet comparitorPSet =
      cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName =
      comparitorPSet.getParameter<std::string>("ComponentName");
  if (comparitorName != "none") {
    theComparitor.reset(SeedComparitorFactory::get()->create(
        comparitorName, comparitorPSet, iC));
  }

}

void CAHitQuadrupletGeneratorGPU::fillDescriptions(
    edm::ParameterSetDescription &desc) {
  desc.add<double>("extraHitRPhitolerance", 0.1);
  desc.add<bool>("fitFastCircle", false);
  desc.add<bool>("fitFastCircleChi2Cut", false);
  desc.add<bool>("useBendingCorrection", false);
  desc.add<double>("CAThetaCut", 0.00125);
  desc.add<double>("CAPhiCut", 10);
  desc.add<double>("CAHardPtCut", 0);
  desc.addOptional<bool>("CAOnlyOneLastHitPerLayerFilter")
      ->setComment(
          "Deprecated and has no effect. To be fully removed later when the "
          "parameter is no longer used in HLT configurations.");
  edm::ParameterSetDescription descMaxChi2;
  descMaxChi2.add<double>("pt1", 0.2);
  descMaxChi2.add<double>("pt2", 1.5);
  descMaxChi2.add<double>("value1", 500);
  descMaxChi2.add<double>("value2", 50);
  descMaxChi2.add<bool>("enabled", true);
  desc.add<edm::ParameterSetDescription>("maxChi2", descMaxChi2);

  edm::ParameterSetDescription descComparitor;
  descComparitor.add<std::string>("ComponentName", "none");
  descComparitor.setAllowAnything(); // until we have moved SeedComparitor too
                                     // to EDProducers
  desc.add<edm::ParameterSetDescription>("SeedComparitorPSet", descComparitor);
}

void CAHitQuadrupletGeneratorGPU::initEvent(const edm::Event &ev,
                                            const edm::EventSetup &es) {
  if (theComparitor)
    theComparitor->init(ev, es);
}

CAHitQuadrupletGeneratorGPU::~CAHitQuadrupletGeneratorGPU() {
    deallocateOnGPU();
}

namespace {
void createGraphStructure(const SeedingLayerSetsHits &layers, CAGraph &g,
                          GPULayerHits *h_layers_, unsigned int maxNumberOfHits_,
                          float *h_x_, float *h_y_, float *h_z_) {
  for (unsigned int i = 0; i < layers.size(); i++) {
    for (unsigned int j = 0; j < 4; ++j) {
      auto vertexIndex = 0;
      auto foundVertex = std::find(g.theLayers.begin(), g.theLayers.end(),
                                   layers[i][j].name());
      if (foundVertex == g.theLayers.end()) {
        g.theLayers.emplace_back(layers[i][j].name(),
                                 layers[i][j].hits().size());
        vertexIndex = g.theLayers.size() - 1;

      } else {
        vertexIndex = foundVertex - g.theLayers.begin();
      }
      if (j == 0) {

        if (std::find(g.theRootLayers.begin(), g.theRootLayers.end(),
                      vertexIndex) == g.theRootLayers.end()) {
          g.theRootLayers.emplace_back(vertexIndex);
        }
      }
    }
  }
}
void clearGraphStructure(const SeedingLayerSetsHits &layers, CAGraph &g) {
  g.theLayerPairs.clear();
  for (unsigned int i = 0; i < g.theLayers.size(); i++) {
    g.theLayers[i].theInnerLayers.clear();
    g.theLayers[i].theInnerLayerPairs.clear();
    g.theLayers[i].theOuterLayers.clear();
    g.theLayers[i].theOuterLayerPairs.clear();
    for (auto &v : g.theLayers[i].isOuterHitOfCell)
      v.clear();
  }
}
void fillGraph(const SeedingLayerSetsHits &layers,
               const IntermediateHitDoublets::RegionLayerSets &regionLayerPairs,
               CAGraph &g, std::vector<const HitDoublets *> &hitDoublets) {

  for (unsigned int i = 0; i < layers.size(); i++) {
    for (unsigned int j = 0; j < 4; ++j) {
      auto vertexIndex = 0;
      auto foundVertex = std::find(g.theLayers.begin(), g.theLayers.end(),
                                   layers[i][j].name());
      if (foundVertex == g.theLayers.end()) {
        vertexIndex = g.theLayers.size() - 1;
      } else {
        vertexIndex = foundVertex - g.theLayers.begin();
      }

      if (j > 0) {
        auto innerVertex = std::find(g.theLayers.begin(), g.theLayers.end(),
                                     layers[i][j - 1].name());

        CALayerPair tmpInnerLayerPair(innerVertex - g.theLayers.begin(),
                                      vertexIndex);

        if (std::find(g.theLayerPairs.begin(), g.theLayerPairs.end(),
                      tmpInnerLayerPair) == g.theLayerPairs.end()) {
          auto found = std::find_if(
              regionLayerPairs.begin(), regionLayerPairs.end(),
              [&](const IntermediateHitDoublets::LayerPairHitDoublets &pair) {
                return pair.innerLayerIndex() == layers[i][j - 1].index() &&
                       pair.outerLayerIndex() == layers[i][j].index();
              });
          if (found != regionLayerPairs.end()) {
            hitDoublets.emplace_back(&(found->doublets()));
            g.theLayerPairs.push_back(tmpInnerLayerPair);
            g.theLayers[vertexIndex].theInnerLayers.push_back(
                innerVertex - g.theLayers.begin());
            innerVertex->theOuterLayers.push_back(vertexIndex);
            g.theLayers[vertexIndex].theInnerLayerPairs.push_back(
                g.theLayerPairs.size() - 1);
            innerVertex->theOuterLayerPairs.push_back(g.theLayerPairs.size() -
                                                      1);
          }
        }
      }
    }
  }

}
} // namespace

void CAHitQuadrupletGeneratorGPU::hitNtuplets(
    const IntermediateHitDoublets &regionDoublets,
    const edm::EventSetup &es,
    const SeedingLayerSetsHits &layers, cudaStream_t cudaStream) {
  CAGraph g;

  hitDoublets.resize(regionDoublets.regionSize());

  for (unsigned int lpIdx = 0; lpIdx < maxNumberOfLayerPairs_; ++lpIdx) {
    h_doublets_[lpIdx].size = 0;
  }
  numberOfRootLayerPairs_ = 0;
  numberOfLayerPairs_ = 0;
  numberOfLayers_ = 0;

  for (unsigned int layerIdx = 0; layerIdx < maxNumberOfLayers_; ++layerIdx) {
    h_layers_[layerIdx].size = 0;
  }

  int index = 0;
  for (const auto &regionLayerPairs : regionDoublets) {
    const TrackingRegion &region = regionLayerPairs.region();
    hitDoublets[index].clear();
    if (index == 0) {
      createGraphStructure(layers, g, h_layers_, maxNumberOfHits_, h_x_, h_y_, h_z_);
    } else {
      clearGraphStructure(layers, g);
    }

    fillGraph(layers, regionLayerPairs, g, hitDoublets[index]);
    numberOfLayers_ = g.theLayers.size();
    numberOfLayerPairs_ = hitDoublets[index].size();
    std::vector<bool> layerAlreadyParsed(g.theLayers.size(), false);

    for (unsigned int i = 0; i < numberOfLayerPairs_; ++i) {
      h_doublets_[i].size = hitDoublets[index][i]->size();
      h_doublets_[i].innerLayerId = g.theLayerPairs[i].theLayers[0];
      h_doublets_[i].outerLayerId = g.theLayerPairs[i].theLayers[1];

      if (layerAlreadyParsed[h_doublets_[i].innerLayerId] == false) {
        layerAlreadyParsed[h_doublets_[i].innerLayerId] = true;
        h_layers_[h_doublets_[i].innerLayerId].size =
            hitDoublets[index][i]->innerLayer().hits().size();
        h_layers_[h_doublets_[i].innerLayerId].layerId =
            h_doublets_[i].innerLayerId;

        for (unsigned int l = 0; l < h_layers_[h_doublets_[i].innerLayerId].size;
             ++l) {
          auto hitId =
              h_layers_[h_doublets_[i].innerLayerId].layerId * maxNumberOfHits_ +
              l;
          h_x_[hitId] =
              hitDoublets[index][i]->innerLayer().hits()[l]->globalPosition().x();
          h_y_[hitId] =
              hitDoublets[index][i]->innerLayer().hits()[l]->globalPosition().y();
          h_z_[hitId] =
              hitDoublets[index][i]->innerLayer().hits()[l]->globalPosition().z();
        }
      }
      if (layerAlreadyParsed[h_doublets_[i].outerLayerId] == false) {
        layerAlreadyParsed[h_doublets_[i].outerLayerId] = true;
        h_layers_[h_doublets_[i].outerLayerId].size =
            hitDoublets[index][i]->outerLayer().hits().size();
        h_layers_[h_doublets_[i].outerLayerId].layerId =
            h_doublets_[i].outerLayerId;
        for (unsigned int l = 0; l < h_layers_[h_doublets_[i].outerLayerId].size;
             ++l) {
          auto hitId =
              h_layers_[h_doublets_[i].outerLayerId].layerId * maxNumberOfHits_ +
              l;
          h_x_[hitId] =
              hitDoublets[index][i]->outerLayer().hits()[l]->globalPosition().x();
          h_y_[hitId] =
              hitDoublets[index][i]->outerLayer().hits()[l]->globalPosition().y();
          h_z_[hitId] =
              hitDoublets[index][i]->outerLayer().hits()[l]->globalPosition().z();
        }
      }

      for (unsigned int rl : g.theRootLayers) {
        if (rl == h_doublets_[i].innerLayerId) {
          auto rootlayerPairId = numberOfRootLayerPairs_;
          h_rootLayerPairs_[rootlayerPairId] = i;
          numberOfRootLayerPairs_++;
        }
      }
      auto numberOfDoublets = hitDoublets[index][i]->size();
      if(numberOfDoublets > maxNumberOfDoublets_)
      {
          edm::LogError("CAHitQuadrupletGeneratorGPU")<<" too many doublets: " << numberOfDoublets << " max is " <<  maxNumberOfDoublets_;
          return;
      }
      for (unsigned int l = 0; l < numberOfDoublets; ++l) {
        auto hitId = i * maxNumberOfDoublets_ * 2 + 2 * l;
        h_indices_[hitId] = hitDoublets[index][i]->innerHitId(l);
        h_indices_[hitId + 1] = hitDoublets[index][i]->outerHitId(l);
      }
    }



    for (unsigned int j = 0; j < numberOfLayerPairs_; ++j) {
      tmp_layerDoublets_[j] = h_doublets_[j];
      tmp_layerDoublets_[j].indices = &d_indices_[j * maxNumberOfDoublets_ * 2];
      cudaMemcpyAsync(&d_indices_[j * maxNumberOfDoublets_ * 2],
                      &h_indices_[j * maxNumberOfDoublets_ * 2],
                      tmp_layerDoublets_[j].size * 2 * sizeof(int),
                      cudaMemcpyHostToDevice, cudaStream);
    }

    for (unsigned int j = 0; j < numberOfLayers_; ++j) {
        if(h_layers_[j].size > maxNumberOfHits_)
        {
            edm::LogError("CAHitQuadrupletGeneratorGPU")<<" too many hits: " << h_layers_[j].size << " max is " << maxNumberOfHits_;
            return;
        }
      tmp_layers_[j] = h_layers_[j];
      tmp_layers_[j].x = &d_x_[maxNumberOfHits_ * j];

      cudaMemcpyAsync(&d_x_[maxNumberOfHits_ * j], &h_x_[j * maxNumberOfHits_],
                      tmp_layers_[j].size * sizeof(float),
                      cudaMemcpyHostToDevice, cudaStream);

      tmp_layers_[j].y = &d_y_[maxNumberOfHits_ * j];
      cudaMemcpyAsync(&d_y_[maxNumberOfHits_ * j], &h_y_[j * maxNumberOfHits_],
                      tmp_layers_[j].size * sizeof(float),
                      cudaMemcpyHostToDevice, cudaStream);

      tmp_layers_[j].z = &d_z_[maxNumberOfHits_ * j];

      cudaMemcpyAsync(&d_z_[maxNumberOfHits_ * j], &h_z_[j * maxNumberOfHits_],
                      tmp_layers_[j].size * sizeof(float),
                      cudaMemcpyHostToDevice, cudaStream);
    }

    cudaMemcpyAsync(d_rootLayerPairs_, h_rootLayerPairs_,
                    numberOfRootLayerPairs_ * sizeof(unsigned int),
                    cudaMemcpyHostToDevice, cudaStream);
    cudaMemcpyAsync(d_doublets_, tmp_layerDoublets_,
                    numberOfLayerPairs_ * sizeof(GPULayerDoublets),
                    cudaMemcpyHostToDevice, cudaStream);
    cudaMemcpyAsync(d_layers_, tmp_layers_, numberOfLayers_ * sizeof(GPULayerHits),
                    cudaMemcpyHostToDevice, cudaStream);

    launchKernels(region, index, cudaStream);
  }
}

void CAHitQuadrupletGeneratorGPU::fillResults(
    const IntermediateHitDoublets &regionDoublets,
    std::vector<OrderedHitSeeds> &result, const edm::EventSetup &es,
    const SeedingLayerSetsHits &layers, cudaStream_t cudaStream)
{
    int index = 0;

    for (const auto &regionLayerPairs : regionDoublets) {
      const TrackingRegion &region = regionLayerPairs.region();
      auto foundQuads = fetchKernelResult(index, cudaStream);
      unsigned int numberOfFoundQuadruplets = foundQuads.size();
      const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);

      // re-used thoughout
      std::array<float, 4> bc_r;
      std::array<float, 4> bc_z;
      std::array<float, 4> bc_errZ2;
      std::array<GlobalPoint, 4> gps;
      std::array<GlobalError, 4> ges;
      std::array<bool, 4> barrels;
      // Loop over quadruplets
      for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId) {
        auto isBarrel = [](const unsigned id) -> bool {
          return id == PixelSubdetector::PixelBarrel;
        };
        for (unsigned int i = 0; i < 3; ++i) {
          auto layerPair = foundQuads[quadId][i].first;
          auto doubletId = foundQuads[quadId][i].second;

          auto const &ahit =
              hitDoublets[index][layerPair]->hit(doubletId, HitDoublets::inner);
          gps[i] = ahit->globalPosition();
          ges[i] = ahit->globalPositionError();
          barrels[i] = isBarrel(ahit->geographicalId().subdetId());

        }
        auto layerPair = foundQuads[quadId][2].first;
        auto doubletId = foundQuads[quadId][2].second;

        auto const &ahit =
            hitDoublets[index][layerPair]->hit(doubletId, HitDoublets::outer);
        gps[3] = ahit->globalPosition();
        ges[3] = ahit->globalPositionError();
        barrels[3] = isBarrel(ahit->geographicalId().subdetId());

        // TODO:
        // - if we decide to always do the circle fit for 4 hits, we don't
        //   need ThirdHitPredictionFromCircle for the curvature; then we
        //   could remove extraHitRPhitolerance configuration parameter
        ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2],
                                                    extraHitRPhitolerance);
        const float curvature = predictionRPhi.curvature(
            ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
        const float abscurv = std::abs(curvature);
        const float thisMaxChi2 = maxChi2Eval.value(abscurv);
        if (theComparitor) {
          SeedingHitSet tmpTriplet(
              hitDoublets[index][foundQuads[quadId][0].first]->hit(
                  foundQuads[quadId][0].second, HitDoublets::inner),
              hitDoublets[index][foundQuads[quadId][2].first]->hit(
                  foundQuads[quadId][2].second, HitDoublets::inner),
              hitDoublets[index][foundQuads[quadId][2].first]->hit(
                  foundQuads[quadId][2].second, HitDoublets::outer));
          if (!theComparitor->compatible(tmpTriplet)) {
            continue;
          }
        }

        float chi2 = std::numeric_limits<float>::quiet_NaN();
        // TODO: Do we have any use case to not use bending correction?
        if (useBendingCorrection) {
          // Following PixelFitterByConformalMappingAndLine
          const float simpleCot = (gps.back().z() - gps.front().z()) /
                                  (gps.back().perp() - gps.front().perp());
          const float pt = 1.f / PixelRecoUtilities::inversePt(abscurv, es);
          for (int i = 0; i < 4; ++i) {
            const GlobalPoint &point = gps[i];
            const GlobalError &error = ges[i];
            bc_r[i] = sqrt(sqr(point.x() - region.origin().x()) +
                           sqr(point.y() - region.origin().y()));
            bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt, es)(
                bc_r[i]);
            bc_z[i] = point.z() - region.origin().z();
            bc_errZ2[i] =
                (barrels[i]) ? error.czz() : error.rerr(point) * sqr(simpleCot);
          }
          RZLine rzLine(bc_r, bc_z, bc_errZ2, RZLine::ErrZ2_tag());
          chi2 = rzLine.chi2();
        } else {
          RZLine rzLine(gps, ges, barrels);
          chi2 = rzLine.chi2();
        }
        if (edm::isNotFinite(chi2) || chi2 > thisMaxChi2) {
          continue;
        }
        // TODO: Do we have any use case to not use circle fit? Maybe
        // HLT where low-pT inefficiency is not a problem?
        if (fitFastCircle) {
          FastCircleFit c(gps, ges);
          chi2 += c.chi2();
          if (edm::isNotFinite(chi2))
            continue;
          if (fitFastCircleChi2Cut && chi2 > thisMaxChi2)
            continue;
        }
        result[index].emplace_back(
            hitDoublets[index][foundQuads[quadId][0].first]->hit(
                foundQuads[quadId][0].second, HitDoublets::inner),
            hitDoublets[index][foundQuads[quadId][1].first]->hit(
                foundQuads[quadId][1].second, HitDoublets::inner),
            hitDoublets[index][foundQuads[quadId][2].first]->hit(
                foundQuads[quadId][2].second, HitDoublets::inner),
            hitDoublets[index][foundQuads[quadId][2].first]->hit(
                foundQuads[quadId][2].second, HitDoublets::outer));
      }
      index++;
  }
}

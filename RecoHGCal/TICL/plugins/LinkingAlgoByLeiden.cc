#include <cmath>
#include <string>
#include "RecoHGCal/TICL/plugins/LinkingAlgoByLeiden.h"

#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/Common.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

using namespace ticl;

LinkingAlgoByLeiden::LinkingAlgoByLeiden(const edm::ParameterSet &conf) : LinkingAlgoBase(conf) {}

LinkingAlgoByLeiden::~LinkingAlgoByLeiden() {}

void LinkingAlgoByLeiden::linkTracksters(const edm::Handle<std::vector<reco::Track>> tkH,
                                         const edm::ValueMap<float> &tkTime,
                                         const edm::ValueMap<float> &tkTimeErr,
                                         const edm::ValueMap<float> &tkTimeQual,
                                         const std::vector<reco::Muon> &muons,
                                         const edm::Handle<std::vector<Trackster>> tsH,
                                         std::vector<TICLCandidate> &resultLinked,
                                         std::vector<TICLCandidate> &chargedHadronsFromTk) {}

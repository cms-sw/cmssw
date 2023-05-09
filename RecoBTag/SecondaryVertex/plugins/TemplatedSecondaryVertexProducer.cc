#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <boost/iterator/transform_iterator.hpp>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/IfExistsDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Common/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackVertexFinder.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrack.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/CandidatePtrTransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "RecoBTag/SecondaryVertex/interface/VertexFilter.h"
#include "RecoBTag/SecondaryVertex/interface/VertexSorting.h"

#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"

//
// constants, enums and typedefs
//
typedef std::shared_ptr<fastjet::ClusterSequence> ClusterSequencePtr;
typedef std::shared_ptr<fastjet::JetDefinition> JetDefPtr;

using namespace reco;

namespace {
  class VertexInfo : public fastjet::PseudoJet::UserInfoBase {
  public:
    VertexInfo(const int vertexIndex) : m_vertexIndex(vertexIndex) {}

    inline const int vertexIndex() const { return m_vertexIndex; }

  protected:
    int m_vertexIndex;
  };

  template <typename T>
  struct RefToBaseLess {
    inline bool operator()(const edm::RefToBase<T> &r1, const edm::RefToBase<T> &r2) const {
      return r1.id() < r2.id() || (r1.id() == r2.id() && r1.key() < r2.key());
    }
  };
}  // namespace

GlobalVector flightDirection(const reco::Vertex &pv, const reco::Vertex &sv) {
  return GlobalVector(sv.x() - pv.x(), sv.y() - pv.y(), sv.z() - pv.z());
}
GlobalVector flightDirection(const reco::Vertex &pv, const reco::VertexCompositePtrCandidate &sv) {
  return GlobalVector(sv.vertex().x() - pv.x(), sv.vertex().y() - pv.y(), sv.vertex().z() - pv.z());
}
const math::XYZPoint &position(const reco::Vertex &sv) { return sv.position(); }
const math::XYZPoint &position(const reco::VertexCompositePtrCandidate &sv) { return sv.vertex(); }

template <class IPTI, class VTX>
class TemplatedSecondaryVertexProducer : public edm::stream::EDProducer<> {
public:
  explicit TemplatedSecondaryVertexProducer(const edm::ParameterSet &params);
  ~TemplatedSecondaryVertexProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  typedef std::vector<TemplatedSecondaryVertexTagInfo<IPTI, VTX> > Product;
  typedef TemplatedSecondaryVertex<VTX> SecondaryVertex;
  typedef typename IPTI::input_container input_container;
  typedef typename IPTI::input_container::value_type input_item;
  typedef typename std::vector<reco::btag::IndexedTrackData> TrackDataVector;
  void produce(edm::Event &event, const edm::EventSetup &es) override;

private:
  template <class CONTAINER>
  void matchReclusteredJets(const edm::Handle<CONTAINER> &jets,
                            const std::vector<fastjet::PseudoJet> &matchedJets,
                            std::vector<int> &matchedIndices,
                            const std::string &jetType = "");
  void matchGroomedJets(const edm::Handle<edm::View<reco::Jet> > &jets,
                        const edm::Handle<edm::View<reco::Jet> > &matchedJets,
                        std::vector<int> &matchedIndices);
  void matchSubjets(const std::vector<int> &groomedIndices,
                    const edm::Handle<edm::View<reco::Jet> > &groomedJets,
                    const edm::Handle<std::vector<IPTI> > &subjets,
                    std::vector<std::vector<int> > &matchedIndices);
  void matchSubjets(const edm::Handle<edm::View<reco::Jet> > &fatJets,
                    const edm::Handle<std::vector<IPTI> > &subjets,
                    std::vector<std::vector<int> > &matchedIndices);

  const reco::Jet *toJet(const reco::Jet &j) { return &j; }
  const reco::Jet *toJet(const IPTI &j) { return &(*(j.jet())); }

  enum ConstraintType {
    CONSTRAINT_NONE = 0,
    CONSTRAINT_BEAMSPOT,
    CONSTRAINT_PV_BEAMSPOT_SIZE,
    CONSTRAINT_PV_BS_Z_ERRORS_SCALED,
    CONSTRAINT_PV_ERROR_SCALED,
    CONSTRAINT_PV_PRIMARIES_IN_FIT
  };
  static ConstraintType getConstraintType(const std::string &name);

  edm::EDGetTokenT<reco::BeamSpot> token_BeamSpot;
  edm::EDGetTokenT<std::vector<IPTI> > token_trackIPTagInfo;
  reco::btag::SortCriteria sortCriterium;
  TrackSelector trackSelector;
  ConstraintType constraint;
  double constraintScaling;
  edm::ParameterSet vtxRecoPSet;
  bool useGhostTrack;
  bool withPVError;
  double minTrackWeight;
  VertexFilter vertexFilter;
  VertexSorting<SecondaryVertex> vertexSorting;
  bool useExternalSV;
  double extSVDeltaRToJet;
  edm::EDGetTokenT<edm::View<VTX> > token_extSVCollection;
  bool useSVClustering;
  bool useSVMomentum;
  std::string jetAlgorithm;
  double rParam;
  double jetPtMin;
  double ghostRescaling;
  double relPtTolerance;
  bool useFatJets;
  bool useGroomedFatJets;
  edm::EDGetTokenT<edm::View<reco::Jet> > token_fatJets;
  edm::EDGetTokenT<edm::View<reco::Jet> > token_groomedFatJets;
  edm::EDGetTokenT<edm::ValueMap<float> > token_weights;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> token_trackBuilder;

  ClusterSequencePtr fjClusterSeq;
  JetDefPtr fjJetDefinition;

  void markUsedTracks(TrackDataVector &trackData,
                      const input_container &trackRefs,
                      const SecondaryVertex &sv,
                      size_t idx);

  struct SVBuilder {
    SVBuilder(const reco::Vertex &pv, const GlobalVector &direction, bool withPVError, double minTrackWeight)
        : pv(pv), direction(direction), withPVError(withPVError), minTrackWeight(minTrackWeight) {}
    SecondaryVertex operator()(const TransientVertex &sv) const;

    SecondaryVertex operator()(const VTX &sv) const { return SecondaryVertex(pv, sv, direction, withPVError); }

    const Vertex &pv;
    const GlobalVector &direction;
    bool withPVError;
    double minTrackWeight;
  };

  struct SVFilter {
    SVFilter(const VertexFilter &filter, const Vertex &pv, const GlobalVector &direction)
        : filter(filter), pv(pv), direction(direction) {}

    inline bool operator()(const SecondaryVertex &sv) const { return !filter(pv, sv, direction); }

    const VertexFilter &filter;
    const Vertex &pv;
    const GlobalVector &direction;
  };
};
template <class IPTI, class VTX>
typename TemplatedSecondaryVertexProducer<IPTI, VTX>::ConstraintType
TemplatedSecondaryVertexProducer<IPTI, VTX>::getConstraintType(const std::string &name) {
  if (name == "None")
    return CONSTRAINT_NONE;
  else if (name == "BeamSpot")
    return CONSTRAINT_BEAMSPOT;
  else if (name == "BeamSpot+PVPosition")
    return CONSTRAINT_PV_BEAMSPOT_SIZE;
  else if (name == "BeamSpotZ+PVErrorScaledXY")
    return CONSTRAINT_PV_BS_Z_ERRORS_SCALED;
  else if (name == "PVErrorScaled")
    return CONSTRAINT_PV_ERROR_SCALED;
  else if (name == "BeamSpot+PVTracksInFit")
    return CONSTRAINT_PV_PRIMARIES_IN_FIT;
  else
    throw cms::Exception("InvalidArgument") << "TemplatedSecondaryVertexProducer: ``constraint'' parameter "
                                               "value \""
                                            << name << "\" not understood." << std::endl;
}

static GhostTrackVertexFinder::FitType getGhostTrackFitType(const std::string &name) {
  if (name == "AlwaysWithGhostTrack")
    return GhostTrackVertexFinder::kAlwaysWithGhostTrack;
  else if (name == "SingleTracksWithGhostTrack")
    return GhostTrackVertexFinder::kSingleTracksWithGhostTrack;
  else if (name == "RefitGhostTrackWithVertices")
    return GhostTrackVertexFinder::kRefitGhostTrackWithVertices;
  else
    throw cms::Exception("InvalidArgument") << "TemplatedSecondaryVertexProducer: ``fitType'' "
                                               "parameter value \""
                                            << name
                                            << "\" for "
                                               "GhostTrackVertexFinder settings not "
                                               "understood."
                                            << std::endl;
}

template <class IPTI, class VTX>
TemplatedSecondaryVertexProducer<IPTI, VTX>::TemplatedSecondaryVertexProducer(const edm::ParameterSet &params)
    : sortCriterium(TrackSorting::getCriterium(params.getParameter<std::string>("trackSort"))),
      trackSelector(params.getParameter<edm::ParameterSet>("trackSelection")),
      constraint(getConstraintType(params.getParameter<std::string>("constraint"))),
      constraintScaling(1.0),
      vtxRecoPSet(params.getParameter<edm::ParameterSet>("vertexReco")),
      useGhostTrack(vtxRecoPSet.getParameter<std::string>("finder") == "gtvr"),
      withPVError(params.getParameter<bool>("usePVError")),
      minTrackWeight(params.getParameter<double>("minimumTrackWeight")),
      vertexFilter(params.getParameter<edm::ParameterSet>("vertexCuts")),
      vertexSorting(params.getParameter<edm::ParameterSet>("vertexSelection")) {
  token_trackIPTagInfo = consumes<std::vector<IPTI> >(params.getParameter<edm::InputTag>("trackIPTagInfos"));
  if (constraint == CONSTRAINT_PV_ERROR_SCALED || constraint == CONSTRAINT_PV_BS_Z_ERRORS_SCALED)
    constraintScaling = params.getParameter<double>("pvErrorScaling");

  if (constraint == CONSTRAINT_PV_BEAMSPOT_SIZE || constraint == CONSTRAINT_PV_BS_Z_ERRORS_SCALED ||
      constraint == CONSTRAINT_BEAMSPOT || constraint == CONSTRAINT_PV_PRIMARIES_IN_FIT)
    token_BeamSpot = consumes<reco::BeamSpot>(params.getParameter<edm::InputTag>("beamSpotTag"));
  useExternalSV = params.getParameter<bool>("useExternalSV");
  if (useExternalSV) {
    token_extSVCollection = consumes<edm::View<VTX> >(params.getParameter<edm::InputTag>("extSVCollection"));
    extSVDeltaRToJet = params.getParameter<double>("extSVDeltaRToJet");
  }
  useSVClustering = (params.existsAs<bool>("useSVClustering") ? params.getParameter<bool>("useSVClustering") : false);
  useSVMomentum = (params.existsAs<bool>("useSVMomentum") ? params.getParameter<bool>("useSVMomentum") : false);
  useFatJets = (useExternalSV && params.exists("fatJets"));
  useGroomedFatJets = (useExternalSV && params.exists("groomedFatJets"));
  if (useSVClustering) {
    jetAlgorithm = params.getParameter<std::string>("jetAlgorithm");
    rParam = params.getParameter<double>("rParam");
    jetPtMin =
        0.;  // hardcoded to 0. since we simply want to recluster all input jets which already had some PtMin applied
    ghostRescaling =
        (params.existsAs<double>("ghostRescaling") ? params.getParameter<double>("ghostRescaling") : 1e-18);
    relPtTolerance =
        (params.existsAs<double>("relPtTolerance")
             ? params.getParameter<double>("relPtTolerance")
             : 1e-03);  // 0.1% relative difference in Pt should be sufficient to detect possible misconfigurations

    // set jet algorithm
    if (jetAlgorithm == "Kt")
      fjJetDefinition = std::make_shared<fastjet::JetDefinition>(fastjet::kt_algorithm, rParam);
    else if (jetAlgorithm == "CambridgeAachen")
      fjJetDefinition = std::make_shared<fastjet::JetDefinition>(fastjet::cambridge_algorithm, rParam);
    else if (jetAlgorithm == "AntiKt")
      fjJetDefinition = std::make_shared<fastjet::JetDefinition>(fastjet::antikt_algorithm, rParam);
    else
      throw cms::Exception("InvalidJetAlgorithm") << "Jet clustering algorithm is invalid: " << jetAlgorithm
                                                  << ", use CambridgeAachen | Kt | AntiKt" << std::endl;
  }
  token_trackBuilder =
      esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"));
  if (useFatJets) {
    token_fatJets = consumes<edm::View<reco::Jet> >(params.getParameter<edm::InputTag>("fatJets"));
  }
  edm::InputTag srcWeights = params.getParameter<edm::InputTag>("weights");
  if (!srcWeights.label().empty())
    token_weights = consumes<edm::ValueMap<float> >(srcWeights);
  if (useGroomedFatJets) {
    token_groomedFatJets = consumes<edm::View<reco::Jet> >(params.getParameter<edm::InputTag>("groomedFatJets"));
  }
  if (useFatJets && !useSVClustering)
    rParam = params.getParameter<double>("rParam");  // will be used later as a dR cut

  produces<Product>();
}
template <class IPTI, class VTX>
TemplatedSecondaryVertexProducer<IPTI, VTX>::~TemplatedSecondaryVertexProducer() {}

template <class IPTI, class VTX>
void TemplatedSecondaryVertexProducer<IPTI, VTX>::produce(edm::Event &event, const edm::EventSetup &es) {
  //	typedef std::map<TrackBaseRef, TransientTrack,
  //	                 RefToBaseLess<Track> > TransientTrackMap;
  //How about good old pointers?
  typedef std::map<const Track *, TransientTrack> TransientTrackMap;

  edm::ESHandle<TransientTrackBuilder> trackBuilder = es.getHandle(token_trackBuilder);

  edm::Handle<std::vector<IPTI> > trackIPTagInfos;
  event.getByToken(token_trackIPTagInfo, trackIPTagInfos);

  // External Sec Vertex collection (e.g. for IVF usage)
  edm::Handle<edm::View<VTX> > extSecVertex;
  if (useExternalSV)
    event.getByToken(token_extSVCollection, extSecVertex);

  edm::Handle<edm::View<reco::Jet> > fatJetsHandle;
  edm::Handle<edm::View<reco::Jet> > groomedFatJetsHandle;
  if (useFatJets) {
    event.getByToken(token_fatJets, fatJetsHandle);
    if (useGroomedFatJets) {
      event.getByToken(token_groomedFatJets, groomedFatJetsHandle);

      if (groomedFatJetsHandle->size() > fatJetsHandle->size())
        edm::LogError("TooManyGroomedJets")
            << "There are more groomed (" << groomedFatJetsHandle->size() << ") than original fat jets ("
            << fatJetsHandle->size() << "). Please check that the two jet collections belong to each other.";
    }
  }
  edm::Handle<edm::ValueMap<float> > weightsHandle;
  if (!token_weights.isUninitialized())
    event.getByToken(token_weights, weightsHandle);

  edm::Handle<BeamSpot> beamSpot;
  unsigned int bsCovSrc[7] = {
      0,
  };
  double sigmaZ = 0.0, beamWidth = 0.0;
  switch (constraint) {
    case CONSTRAINT_PV_BEAMSPOT_SIZE:
      event.getByToken(token_BeamSpot, beamSpot);
      bsCovSrc[3] = bsCovSrc[4] = bsCovSrc[5] = bsCovSrc[6] = 1;
      sigmaZ = beamSpot->sigmaZ();
      beamWidth = beamSpot->BeamWidthX();
      break;

    case CONSTRAINT_PV_BS_Z_ERRORS_SCALED:
      event.getByToken(token_BeamSpot, beamSpot);
      bsCovSrc[0] = bsCovSrc[1] = 2;
      bsCovSrc[3] = bsCovSrc[4] = bsCovSrc[5] = 1;
      sigmaZ = beamSpot->sigmaZ();
      break;

    case CONSTRAINT_PV_ERROR_SCALED:
      bsCovSrc[0] = bsCovSrc[1] = bsCovSrc[2] = 2;
      break;

    case CONSTRAINT_BEAMSPOT:
    case CONSTRAINT_PV_PRIMARIES_IN_FIT:
      event.getByToken(token_BeamSpot, beamSpot);
      break;

    default:
        /* nothing */;
  }

  // ------------------------------------ SV clustering START --------------------------------------------
  std::vector<std::vector<int> > clusteredSVs(trackIPTagInfos->size(), std::vector<int>());
  if (useExternalSV && useSVClustering && !trackIPTagInfos->empty()) {
    // vector of constituents for reclustering jets and "ghost" SVs
    std::vector<fastjet::PseudoJet> fjInputs;
    // loop over all input jets and collect all their constituents
    if (useFatJets) {
      for (edm::View<reco::Jet>::const_iterator it = fatJetsHandle->begin(); it != fatJetsHandle->end(); ++it) {
        std::vector<edm::Ptr<reco::Candidate> > constituents = it->getJetConstituents();
        std::vector<edm::Ptr<reco::Candidate> >::const_iterator m;
        for (m = constituents.begin(); m != constituents.end(); ++m) {
          const reco::CandidatePtr &constit = *m;
          if (constit.isNull() || constit->pt() <= std::numeric_limits<double>::epsilon()) {
            edm::LogWarning("NullTransverseMomentum") << "dropping input candidate with pt=0";
            continue;
          }
          if (it->isWeighted()) {
            if (token_weights.isUninitialized())
              throw cms::Exception("MissingConstituentWeight")
                  << "TemplatedSecondaryVertexProducer: No weights (e.g. PUPPI) given for weighted jet collection"
                  << std::endl;
            float w = (*weightsHandle)[constit];
            if (w > 0) {
              fjInputs.push_back(
                  fastjet::PseudoJet(constit->px() * w, constit->py() * w, constit->pz() * w, constit->energy() * w));
            }
          } else {
            fjInputs.push_back(fastjet::PseudoJet(constit->px(), constit->py(), constit->pz(), constit->energy()));
          }
        }
      }
    } else {
      for (typename std::vector<IPTI>::const_iterator it = trackIPTagInfos->begin(); it != trackIPTagInfos->end();
           ++it) {
        std::vector<edm::Ptr<reco::Candidate> > constituents = it->jet()->getJetConstituents();
        std::vector<edm::Ptr<reco::Candidate> >::const_iterator m;
        for (m = constituents.begin(); m != constituents.end(); ++m) {
          const reco::CandidatePtr &constit = *m;
          if (constit.isNull() || constit->pt() <= std::numeric_limits<double>::epsilon()) {
            edm::LogWarning("NullTransverseMomentum") << "dropping input candidate with pt=0";
            continue;
          }
          if (it->jet()->isWeighted()) {
            if (token_weights.isUninitialized())
              throw cms::Exception("MissingConstituentWeight")
                  << "TemplatedSecondaryVertexProducer: No weights (e.g. PUPPI) given for weighted jet collection"
                  << std::endl;
            float w = (*weightsHandle)[constit];
            if (w > 0) {
              fjInputs.push_back(
                  fastjet::PseudoJet(constit->px() * w, constit->py() * w, constit->pz() * w, constit->energy() * w));
            }
          } else {
            fjInputs.push_back(fastjet::PseudoJet(constit->px(), constit->py(), constit->pz(), constit->energy()));
          }
        }
      }
    }
    // insert "ghost" SVs in the vector of constituents
    for (typename edm::View<VTX>::const_iterator it = extSecVertex->begin(); it != extSecVertex->end(); ++it) {
      const reco::Vertex &pv = *(trackIPTagInfos->front().primaryVertex());
      GlobalVector dir = flightDirection(pv, *it);
      dir = dir.unit();
      fastjet::PseudoJet p(
          dir.x(), dir.y(), dir.z(), dir.mag());  // using SV flight direction so treating SV as massless
      if (useSVMomentum)
        p = fastjet::PseudoJet(it->p4().px(), it->p4().py(), it->p4().pz(), it->p4().energy());
      p *= ghostRescaling;  // rescale SV direction/momentum
      p.set_user_info(new VertexInfo(it - extSecVertex->begin()));
      fjInputs.push_back(p);
    }

    // define jet clustering sequence
    fjClusterSeq = std::make_shared<fastjet::ClusterSequence>(fjInputs, *fjJetDefinition);
    // recluster jet constituents and inserted "ghosts"
    std::vector<fastjet::PseudoJet> inclusiveJets = fastjet::sorted_by_pt(fjClusterSeq->inclusive_jets(jetPtMin));

    if (useFatJets) {
      if (inclusiveJets.size() < fatJetsHandle->size())
        edm::LogError("TooFewReclusteredJets")
            << "There are fewer reclustered (" << inclusiveJets.size() << ") than original fat jets ("
            << fatJetsHandle->size()
            << "). Please check that the jet algorithm and jet size match those used for the original jet collection.";

      // match reclustered and original fat jets
      std::vector<int> reclusteredIndices;
      matchReclusteredJets<edm::View<reco::Jet> >(fatJetsHandle, inclusiveJets, reclusteredIndices, "fat");

      // match groomed and original fat jets
      std::vector<int> groomedIndices;
      if (useGroomedFatJets)
        matchGroomedJets(fatJetsHandle, groomedFatJetsHandle, groomedIndices);

      // match subjets and original fat jets
      std::vector<std::vector<int> > subjetIndices;
      if (useGroomedFatJets)
        matchSubjets(groomedIndices, groomedFatJetsHandle, trackIPTagInfos, subjetIndices);
      else
        matchSubjets(fatJetsHandle, trackIPTagInfos, subjetIndices);

      // collect clustered SVs
      for (size_t i = 0; i < fatJetsHandle->size(); ++i) {
        if (reclusteredIndices.at(i) < 0)
          continue;  // continue if matching reclustered to original jets failed

        if (fatJetsHandle->at(i).pt() == 0)  // continue if the original jet has Pt=0
        {
          edm::LogWarning("NullTransverseMomentum")
              << "The original fat jet " << i << " has Pt=0. This is not expected so the jet will be skipped.";
          continue;
        }

        if (subjetIndices.at(i).empty())
          continue;  // continue if the original jet does not have subjets assigned

        // since the "ghosts" are extremely soft, the configuration and ordering of the reclustered and original fat jets should in principle stay the same
        if ((std::abs(inclusiveJets.at(reclusteredIndices.at(i)).pt() - fatJetsHandle->at(i).pt()) /
             fatJetsHandle->at(i).pt()) > relPtTolerance) {
          if (fatJetsHandle->at(i).pt() < 10.)  // special handling for low-Pt jets (Pt<10 GeV)
            edm::LogWarning("JetPtMismatchAtLowPt")
                << "The reclustered and original fat jet " << i << " have different Pt's ("
                << inclusiveJets.at(reclusteredIndices.at(i)).pt() << " vs " << fatJetsHandle->at(i).pt()
                << " GeV, respectively).\n"
                << "Please check that the jet algorithm and jet size match those used for the original fat jet "
                   "collection and also make sure the original fat jets are uncorrected. In addition, make sure you "
                   "are not using CaloJets which are presently not supported.\n"
                << "Since the mismatch is at low Pt, it is ignored and only a warning is issued.\n"
                << "\nIn extremely rare instances the mismatch could be caused by a difference in the machine "
                   "precision in which case make sure the original jet collection is produced and reclustering is "
                   "performed in the same job.";
          else
            edm::LogError("JetPtMismatch")
                << "The reclustered and original fat jet " << i << " have different Pt's ("
                << inclusiveJets.at(reclusteredIndices.at(i)).pt() << " vs " << fatJetsHandle->at(i).pt()
                << " GeV, respectively).\n"
                << "Please check that the jet algorithm and jet size match those used for the original fat jet "
                   "collection and also make sure the original fat jets are uncorrected. In addition, make sure you "
                   "are not using CaloJets which are presently not supported.\n"
                << "\nIn extremely rare instances the mismatch could be caused by a difference in the machine "
                   "precision in which case make sure the original jet collection is produced and reclustering is "
                   "performed in the same job.";
        }

        // get jet constituents
        std::vector<fastjet::PseudoJet> constituents = inclusiveJets.at(reclusteredIndices.at(i)).constituents();

        std::vector<int> svIndices;
        // loop over jet constituents and try to find "ghosts"
        for (std::vector<fastjet::PseudoJet>::const_iterator it = constituents.begin(); it != constituents.end();
             ++it) {
          if (!it->has_user_info())
            continue;  // skip if not a "ghost"

          svIndices.push_back(it->user_info<VertexInfo>().vertexIndex());
        }

        // loop over clustered SVs and assign them to different subjets based on smallest dR
        for (size_t sv = 0; sv < svIndices.size(); ++sv) {
          const reco::Vertex &pv = *(trackIPTagInfos->front().primaryVertex());
          const VTX &extSV = (*extSecVertex)[svIndices.at(sv)];
          GlobalVector dir = flightDirection(pv, extSV);
          dir = dir.unit();
          fastjet::PseudoJet p(
              dir.x(), dir.y(), dir.z(), dir.mag());  // using SV flight direction so treating SV as massless
          if (useSVMomentum)
            p = fastjet::PseudoJet(extSV.p4().px(), extSV.p4().py(), extSV.p4().pz(), extSV.p4().energy());

          std::vector<double> dR2toSubjets;

          for (size_t sj = 0; sj < subjetIndices.at(i).size(); ++sj)
            dR2toSubjets.push_back(Geom::deltaR2(p.rapidity(),
                                                 p.phi_std(),
                                                 trackIPTagInfos->at(subjetIndices.at(i).at(sj)).jet()->rapidity(),
                                                 trackIPTagInfos->at(subjetIndices.at(i).at(sj)).jet()->phi()));

          // find the closest subjet
          int closestSubjetIdx =
              std::distance(dR2toSubjets.begin(), std::min_element(dR2toSubjets.begin(), dR2toSubjets.end()));

          clusteredSVs.at(subjetIndices.at(i).at(closestSubjetIdx)).push_back(svIndices.at(sv));
        }
      }
    } else {
      if (inclusiveJets.size() < trackIPTagInfos->size())
        edm::LogError("TooFewReclusteredJets")
            << "There are fewer reclustered (" << inclusiveJets.size() << ") than original jets ("
            << trackIPTagInfos->size()
            << "). Please check that the jet algorithm and jet size match those used for the original jet collection.";

      // match reclustered and original jets
      std::vector<int> reclusteredIndices;
      matchReclusteredJets<std::vector<IPTI> >(trackIPTagInfos, inclusiveJets, reclusteredIndices);

      // collect clustered SVs
      for (size_t i = 0; i < trackIPTagInfos->size(); ++i) {
        if (reclusteredIndices.at(i) < 0)
          continue;  // continue if matching reclustered to original jets failed

        if (trackIPTagInfos->at(i).jet()->pt() == 0)  // continue if the original jet has Pt=0
        {
          edm::LogWarning("NullTransverseMomentum")
              << "The original jet " << i << " has Pt=0. This is not expected so the jet will be skipped.";
          continue;
        }

        // since the "ghosts" are extremely soft, the configuration and ordering of the reclustered and original jets should in principle stay the same
        if ((std::abs(inclusiveJets.at(reclusteredIndices.at(i)).pt() - trackIPTagInfos->at(i).jet()->pt()) /
             trackIPTagInfos->at(i).jet()->pt()) > relPtTolerance) {
          if (trackIPTagInfos->at(i).jet()->pt() < 10.)  // special handling for low-Pt jets (Pt<10 GeV)
            edm::LogWarning("JetPtMismatchAtLowPt")
                << "The reclustered and original jet " << i << " have different Pt's ("
                << inclusiveJets.at(reclusteredIndices.at(i)).pt() << " vs " << trackIPTagInfos->at(i).jet()->pt()
                << " GeV, respectively).\n"
                << "Please check that the jet algorithm and jet size match those used for the original jet collection "
                   "and also make sure the original jets are uncorrected. In addition, make sure you are not using "
                   "CaloJets which are presently not supported.\n"
                << "Since the mismatch is at low Pt, it is ignored and only a warning is issued.\n"
                << "\nIn extremely rare instances the mismatch could be caused by a difference in the machine "
                   "precision in which case make sure the original jet collection is produced and reclustering is "
                   "performed in the same job.";
          else
            edm::LogError("JetPtMismatch")
                << "The reclustered and original jet " << i << " have different Pt's ("
                << inclusiveJets.at(reclusteredIndices.at(i)).pt() << " vs " << trackIPTagInfos->at(i).jet()->pt()
                << " GeV, respectively).\n"
                << "Please check that the jet algorithm and jet size match those used for the original jet collection "
                   "and also make sure the original jets are uncorrected. In addition, make sure you are not using "
                   "CaloJets which are presently not supported.\n"
                << "\nIn extremely rare instances the mismatch could be caused by a difference in the machine "
                   "precision in which case make sure the original jet collection is produced and reclustering is "
                   "performed in the same job.";
        }

        // get jet constituents
        std::vector<fastjet::PseudoJet> constituents = inclusiveJets.at(reclusteredIndices.at(i)).constituents();

        // loop over jet constituents and try to find "ghosts"
        for (std::vector<fastjet::PseudoJet>::const_iterator it = constituents.begin(); it != constituents.end();
             ++it) {
          if (!it->has_user_info())
            continue;  // skip if not a "ghost"
          // push back clustered SV indices
          clusteredSVs.at(i).push_back(it->user_info<VertexInfo>().vertexIndex());
        }
      }
    }
  }
  // case where fat jets are used to associate SVs to subjets but no SV clustering is performed
  else if (useExternalSV && !useSVClustering && !trackIPTagInfos->empty() && useFatJets) {
    // match groomed and original fat jets
    std::vector<int> groomedIndices;
    if (useGroomedFatJets)
      matchGroomedJets(fatJetsHandle, groomedFatJetsHandle, groomedIndices);

    // match subjets and original fat jets
    std::vector<std::vector<int> > subjetIndices;
    if (useGroomedFatJets)
      matchSubjets(groomedIndices, groomedFatJetsHandle, trackIPTagInfos, subjetIndices);
    else
      matchSubjets(fatJetsHandle, trackIPTagInfos, subjetIndices);

    // loop over fat jets
    for (size_t i = 0; i < fatJetsHandle->size(); ++i) {
      if (fatJetsHandle->at(i).pt() == 0)  // continue if the original jet has Pt=0
      {
        edm::LogWarning("NullTransverseMomentum")
            << "The original fat jet " << i << " has Pt=0. This is not expected so the jet will be skipped.";
        continue;
      }

      if (subjetIndices.at(i).empty())
        continue;  // continue if the original jet does not have subjets assigned

      // loop over SVs, associate them to fat jets based on dR cone and
      // then assign them to the closets subjet in dR
      for (typename edm::View<VTX>::const_iterator it = extSecVertex->begin(); it != extSecVertex->end(); ++it) {
        size_t sv = (it - extSecVertex->begin());

        const reco::Vertex &pv = *(trackIPTagInfos->front().primaryVertex());
        const VTX &extSV = (*extSecVertex)[sv];
        GlobalVector dir = flightDirection(pv, extSV);
        GlobalVector jetDir(fatJetsHandle->at(i).px(), fatJetsHandle->at(i).py(), fatJetsHandle->at(i).pz());
        // skip SVs outside the dR cone
        if (Geom::deltaR2(dir, jetDir) > rParam * rParam)  // here using the jet clustering rParam as a dR cut
          continue;

        dir = dir.unit();
        fastjet::PseudoJet p(
            dir.x(), dir.y(), dir.z(), dir.mag());  // using SV flight direction so treating SV as massless
        if (useSVMomentum)
          p = fastjet::PseudoJet(extSV.p4().px(), extSV.p4().py(), extSV.p4().pz(), extSV.p4().energy());

        std::vector<double> dR2toSubjets;

        for (size_t sj = 0; sj < subjetIndices.at(i).size(); ++sj)
          dR2toSubjets.push_back(Geom::deltaR2(p.rapidity(),
                                               p.phi_std(),
                                               trackIPTagInfos->at(subjetIndices.at(i).at(sj)).jet()->rapidity(),
                                               trackIPTagInfos->at(subjetIndices.at(i).at(sj)).jet()->phi()));

        // find the closest subjet
        int closestSubjetIdx =
            std::distance(dR2toSubjets.begin(), std::min_element(dR2toSubjets.begin(), dR2toSubjets.end()));

        clusteredSVs.at(subjetIndices.at(i).at(closestSubjetIdx)).push_back(sv);
      }
    }
  }
  // ------------------------------------ SV clustering END ----------------------------------------------

  std::unique_ptr<ConfigurableVertexReconstructor> vertexReco;
  std::unique_ptr<GhostTrackVertexFinder> vertexRecoGT;
  if (useGhostTrack)
    vertexRecoGT = std::make_unique<GhostTrackVertexFinder>(
        vtxRecoPSet.getParameter<double>("maxFitChi2"),
        vtxRecoPSet.getParameter<double>("mergeThreshold"),
        vtxRecoPSet.getParameter<double>("primcut"),
        vtxRecoPSet.getParameter<double>("seccut"),
        getGhostTrackFitType(vtxRecoPSet.getParameter<std::string>("fitType")));
  else
    vertexReco = std::make_unique<ConfigurableVertexReconstructor>(vtxRecoPSet);

  TransientTrackMap primariesMap;

  // result secondary vertices

  auto tagInfos = std::make_unique<Product>();

  for (typename std::vector<IPTI>::const_iterator iterJets = trackIPTagInfos->begin();
       iterJets != trackIPTagInfos->end();
       ++iterJets) {
    TrackDataVector trackData;
    //		      std::cout << "Jet " << iterJets-trackIPTagInfos->begin() << std::endl;

    const Vertex &pv = *iterJets->primaryVertex();

    std::set<TransientTrack> primaries;
    if (constraint == CONSTRAINT_PV_PRIMARIES_IN_FIT) {
      for (Vertex::trackRef_iterator iter = pv.tracks_begin(); iter != pv.tracks_end(); ++iter) {
        TransientTrackMap::iterator pos = primariesMap.lower_bound(iter->get());

        if (pos != primariesMap.end() && pos->first == iter->get())
          primaries.insert(pos->second);
        else {
          TransientTrack track = trackBuilder->build(iter->castTo<TrackRef>());
          primariesMap.insert(pos, std::make_pair(iter->get(), track));
          primaries.insert(track);
        }
      }
    }

    edm::RefToBase<Jet> jetRef = iterJets->jet();

    GlobalVector jetDir(jetRef->momentum().x(), jetRef->momentum().y(), jetRef->momentum().z());

    std::vector<std::size_t> indices = iterJets->sortedIndexes(sortCriterium);

    input_container trackRefs = iterJets->sortedTracks(indices);

    const std::vector<reco::btag::TrackIPData> &ipData = iterJets->impactParameterData();

    // build transient tracks used for vertex reconstruction

    std::vector<TransientTrack> fitTracks;
    std::vector<GhostTrackState> gtStates;
    std::unique_ptr<GhostTrackPrediction> gtPred;
    if (useGhostTrack)
      gtPred = std::make_unique<GhostTrackPrediction>(*iterJets->ghostTrack());

    for (unsigned int i = 0; i < indices.size(); i++) {
      typedef typename TemplatedSecondaryVertexTagInfo<IPTI, VTX>::IndexedTrackData IndexedTrackData;

      const input_item &trackRef = trackRefs[i];

      trackData.push_back(IndexedTrackData());
      trackData.back().first = indices[i];

      // select tracks for SV finder

      if (!trackSelector(
              *reco::btag::toTrack(trackRef), ipData[indices[i]], *jetRef, RecoVertex::convertPos(pv.position()))) {
        trackData.back().second.svStatus = TemplatedSecondaryVertexTagInfo<IPTI, VTX>::TrackData::trackSelected;
        continue;
      }

      TransientTrackMap::const_iterator pos = primariesMap.find(reco::btag::toTrack((trackRef)));
      TransientTrack fitTrack;
      if (pos != primariesMap.end()) {
        primaries.erase(pos->second);
        fitTrack = pos->second;
      } else
        fitTrack = trackBuilder->build(trackRef);
      fitTracks.push_back(fitTrack);

      trackData.back().second.svStatus = TemplatedSecondaryVertexTagInfo<IPTI, VTX>::TrackData::trackUsedForVertexFit;

      if (useGhostTrack) {
        GhostTrackState gtState(fitTrack);
        GlobalPoint pos = ipData[indices[i]].closestToGhostTrack;
        gtState.linearize(*gtPred, true, gtPred->lambda(pos));
        gtState.setWeight(ipData[indices[i]].ghostTrackWeight);
        gtStates.push_back(gtState);
      }
    }

    std::unique_ptr<GhostTrack> ghostTrack;
    if (useGhostTrack)
      ghostTrack = std::make_unique<GhostTrack>(
          GhostTrackPrediction(
              RecoVertex::convertPos(pv.position()),
              RecoVertex::convertError(pv.error()),
              GlobalVector(iterJets->ghostTrack()->px(), iterJets->ghostTrack()->py(), iterJets->ghostTrack()->pz()),
              0.05),
          *gtPred,
          gtStates,
          iterJets->ghostTrack()->chi2(),
          iterJets->ghostTrack()->ndof());

    // perform actual vertex finding

    std::vector<VTX> extAssoCollection;
    std::vector<TransientVertex> fittedSVs;
    std::vector<SecondaryVertex> SVs;
    if (!useExternalSV) {
      switch (constraint) {
        case CONSTRAINT_NONE:
          if (useGhostTrack)
            fittedSVs = vertexRecoGT->vertices(pv, *ghostTrack);
          else
            fittedSVs = vertexReco->vertices(fitTracks);
          break;

        case CONSTRAINT_BEAMSPOT:
          if (useGhostTrack)
            fittedSVs = vertexRecoGT->vertices(pv, *beamSpot, *ghostTrack);
          else
            fittedSVs = vertexReco->vertices(fitTracks, *beamSpot);
          break;

        case CONSTRAINT_PV_BEAMSPOT_SIZE:
        case CONSTRAINT_PV_BS_Z_ERRORS_SCALED:
        case CONSTRAINT_PV_ERROR_SCALED: {
          BeamSpot::CovarianceMatrix cov;
          for (unsigned int i = 0; i < 7; i++) {
            unsigned int covSrc = bsCovSrc[i];
            for (unsigned int j = 0; j < 7; j++) {
              double v = 0.0;
              if (!covSrc || bsCovSrc[j] != covSrc)
                v = 0.0;
              else if (covSrc == 1)
                v = beamSpot->covariance(i, j);
              else if (j < 3 && i < 3)
                v = pv.covariance(i, j) * constraintScaling;
              cov(i, j) = v;
            }
          }

          BeamSpot bs(pv.position(),
                      sigmaZ,
                      beamSpot.isValid() ? beamSpot->dxdz() : 0.,
                      beamSpot.isValid() ? beamSpot->dydz() : 0.,
                      beamWidth,
                      cov,
                      BeamSpot::Unknown);

          if (useGhostTrack)
            fittedSVs = vertexRecoGT->vertices(pv, bs, *ghostTrack);
          else
            fittedSVs = vertexReco->vertices(fitTracks, bs);
        } break;

        case CONSTRAINT_PV_PRIMARIES_IN_FIT: {
          std::vector<TransientTrack> primaries_(primaries.begin(), primaries.end());
          if (useGhostTrack)
            fittedSVs = vertexRecoGT->vertices(pv, *beamSpot, primaries_, *ghostTrack);
          else
            fittedSVs = vertexReco->vertices(primaries_, fitTracks, *beamSpot);
        } break;
      }
      // build combined SV information and filter
      SVBuilder svBuilder(pv, jetDir, withPVError, minTrackWeight);
      std::remove_copy_if(boost::make_transform_iterator(fittedSVs.begin(), svBuilder),
                          boost::make_transform_iterator(fittedSVs.end(), svBuilder),
                          std::back_inserter(SVs),
                          SVFilter(vertexFilter, pv, jetDir));

    } else {
      if (useSVClustering || useFatJets) {
        size_t jetIdx = (iterJets - trackIPTagInfos->begin());

        for (size_t iExtSv = 0; iExtSv < clusteredSVs.at(jetIdx).size(); iExtSv++) {
          const VTX &extVertex = (*extSecVertex)[clusteredSVs.at(jetIdx).at(iExtSv)];
          if (extVertex.p4().M() < 0.3)
            continue;
          extAssoCollection.push_back(extVertex);
        }
      } else {
        for (size_t iExtSv = 0; iExtSv < extSecVertex->size(); iExtSv++) {
          const VTX &extVertex = (*extSecVertex)[iExtSv];
          if (Geom::deltaR2((position(extVertex) - pv.position()), (extSVDeltaRToJet > 0) ? jetDir : -jetDir) >
                  extSVDeltaRToJet * extSVDeltaRToJet ||
              extVertex.p4().M() < 0.3)
            continue;
          extAssoCollection.push_back(extVertex);
        }
      }
      // build combined SV information and filter
      SVBuilder svBuilder(pv, jetDir, withPVError, minTrackWeight);
      std::remove_copy_if(boost::make_transform_iterator(extAssoCollection.begin(), svBuilder),
                          boost::make_transform_iterator(extAssoCollection.end(), svBuilder),
                          std::back_inserter(SVs),
                          SVFilter(vertexFilter, pv, jetDir));
    }
    // clean up now unneeded collections
    gtPred.reset();
    ghostTrack.reset();
    gtStates.clear();
    fitTracks.clear();
    fittedSVs.clear();
    extAssoCollection.clear();

    // sort SVs by importance

    std::vector<unsigned int> vtxIndices = vertexSorting(SVs);

    std::vector<typename TemplatedSecondaryVertexTagInfo<IPTI, VTX>::VertexData> svData;

    svData.resize(vtxIndices.size());
    for (unsigned int idx = 0; idx < vtxIndices.size(); idx++) {
      const SecondaryVertex &sv = SVs[vtxIndices[idx]];

      svData[idx].vertex = sv;
      svData[idx].dist1d = sv.dist1d();
      svData[idx].dist2d = sv.dist2d();
      svData[idx].dist3d = sv.dist3d();
      svData[idx].direction = flightDirection(pv, sv);
      // mark tracks successfully used in vertex fit
      markUsedTracks(trackData, trackRefs, sv, idx);
    }

    // fill result into tag infos

    tagInfos->push_back(TemplatedSecondaryVertexTagInfo<IPTI, VTX>(
        trackData,
        svData,
        SVs.size(),
        edm::Ref<std::vector<IPTI> >(trackIPTagInfos, iterJets - trackIPTagInfos->begin())));
  }

  event.put(std::move(tagInfos));
}

//Need specialized template because reco::Vertex iterators are TrackBase and it is a mess to make general
template <>
void TemplatedSecondaryVertexProducer<TrackIPTagInfo, reco::Vertex>::markUsedTracks(TrackDataVector &trackData,
                                                                                    const input_container &trackRefs,
                                                                                    const SecondaryVertex &sv,
                                                                                    size_t idx) {
  for (Vertex::trackRef_iterator iter = sv.tracks_begin(); iter != sv.tracks_end(); ++iter) {
    if (sv.trackWeight(*iter) < minTrackWeight)
      continue;

    typename input_container::const_iterator pos =
        std::find(trackRefs.begin(), trackRefs.end(), iter->castTo<input_item>());

    if (pos == trackRefs.end()) {
      if (!useExternalSV)
        throw cms::Exception("TrackNotFound") << "Could not find track from secondary "
                                                 "vertex in original tracks."
                                              << std::endl;
    } else {
      unsigned int index = pos - trackRefs.begin();
      trackData[index].second.svStatus = (btag::TrackData::trackAssociatedToVertex + idx);
    }
  }
}
template <>
void TemplatedSecondaryVertexProducer<CandIPTagInfo, reco::VertexCompositePtrCandidate>::markUsedTracks(
    TrackDataVector &trackData, const input_container &trackRefs, const SecondaryVertex &sv, size_t idx) {
  for (typename input_container::const_iterator iter = sv.daughterPtrVector().begin();
       iter != sv.daughterPtrVector().end();
       ++iter) {
    typename input_container::const_iterator pos = std::find(trackRefs.begin(), trackRefs.end(), *iter);

    if (pos != trackRefs.end()) {
      unsigned int index = pos - trackRefs.begin();
      trackData[index].second.svStatus = (btag::TrackData::trackAssociatedToVertex + idx);
    }
  }
}

template <>
typename TemplatedSecondaryVertexProducer<TrackIPTagInfo, reco::Vertex>::SecondaryVertex
TemplatedSecondaryVertexProducer<TrackIPTagInfo, reco::Vertex>::SVBuilder::operator()(const TransientVertex &sv) const {
  if (!sv.originalTracks().empty() && sv.originalTracks()[0].trackBaseRef().isNonnull())
    return SecondaryVertex(pv, sv, direction, withPVError);
  else {
    edm::LogError("UnexpectedInputs") << "Building from Candidates, should not happen!";
    return SecondaryVertex(pv, sv, direction, withPVError);
  }
}

template <>
typename TemplatedSecondaryVertexProducer<CandIPTagInfo, reco::VertexCompositePtrCandidate>::SecondaryVertex
TemplatedSecondaryVertexProducer<CandIPTagInfo, reco::VertexCompositePtrCandidate>::SVBuilder::operator()(
    const TransientVertex &sv) const {
  if (!sv.originalTracks().empty() && sv.originalTracks()[0].trackBaseRef().isNonnull()) {
    edm::LogError("UnexpectedInputs") << "Building from Tracks, should not happen!";
    VertexCompositePtrCandidate vtxCompPtrCand;

    vtxCompPtrCand.setCovariance(sv.vertexState().error().matrix());
    vtxCompPtrCand.setChi2AndNdof(sv.totalChiSquared(), sv.degreesOfFreedom());
    vtxCompPtrCand.setVertex(Candidate::Point(sv.position().x(), sv.position().y(), sv.position().z()));

    return SecondaryVertex(pv, vtxCompPtrCand, direction, withPVError);
  } else {
    VertexCompositePtrCandidate vtxCompPtrCand;

    vtxCompPtrCand.setCovariance(sv.vertexState().error().matrix());
    vtxCompPtrCand.setChi2AndNdof(sv.totalChiSquared(), sv.degreesOfFreedom());
    vtxCompPtrCand.setVertex(Candidate::Point(sv.position().x(), sv.position().y(), sv.position().z()));

    Candidate::LorentzVector p4;
    for (std::vector<reco::TransientTrack>::const_iterator tt = sv.originalTracks().begin();
         tt != sv.originalTracks().end();
         ++tt) {
      if (sv.trackWeight(*tt) < minTrackWeight)
        continue;

      const CandidatePtrTransientTrack *cptt =
          dynamic_cast<const CandidatePtrTransientTrack *>(tt->basicTransientTrack());
      if (cptt == nullptr)
        edm::LogError("DynamicCastingFailed") << "Casting of TransientTrack to CandidatePtrTransientTrack failed!";
      else {
        p4 += cptt->candidate()->p4();
        vtxCompPtrCand.addDaughter(cptt->candidate());
      }
    }
    vtxCompPtrCand.setP4(p4);

    return SecondaryVertex(pv, vtxCompPtrCand, direction, withPVError);
  }
}

// ------------ method that matches reclustered and original jets based on minimum dR ------------
template <class IPTI, class VTX>
template <class CONTAINER>
void TemplatedSecondaryVertexProducer<IPTI, VTX>::matchReclusteredJets(
    const edm::Handle<CONTAINER> &jets,
    const std::vector<fastjet::PseudoJet> &reclusteredJets,
    std::vector<int> &matchedIndices,
    const std::string &jetType) {
  std::string type = (!jetType.empty() ? jetType + " " : jetType);

  std::vector<bool> matchedLocks(reclusteredJets.size(), false);

  for (size_t j = 0; j < jets->size(); ++j) {
    double matchedDR2 = 1e9;
    int matchedIdx = -1;

    for (size_t rj = 0; rj < reclusteredJets.size(); ++rj) {
      if (matchedLocks.at(rj))
        continue;  // skip jets that have already been matched

      double tempDR2 = Geom::deltaR2(toJet(jets->at(j))->rapidity(),
                                     toJet(jets->at(j))->phi(),
                                     reclusteredJets.at(rj).rapidity(),
                                     reclusteredJets.at(rj).phi_std());
      if (tempDR2 < matchedDR2) {
        matchedDR2 = tempDR2;
        matchedIdx = rj;
      }
    }

    if (matchedIdx >= 0) {
      if (matchedDR2 > rParam * rParam) {
        edm::LogError("JetMatchingFailed") << "Matched reclustered jet " << matchedIdx << " and original " << type
                                           << "jet " << j << " are separated by dR=" << sqrt(matchedDR2)
                                           << " which is greater than the jet size R=" << rParam << ".\n"
                                           << "This is not expected so please check that the jet algorithm and jet "
                                              "size match those used for the original "
                                           << type << "jet collection.";
      } else
        matchedLocks.at(matchedIdx) = true;
    } else
      edm::LogError("JetMatchingFailed")
          << "Matching reclustered to original " << type
          << "jets failed. Please check that the jet algorithm and jet size match those used for the original " << type
          << "jet collection.";

    matchedIndices.push_back(matchedIdx);
  }
}

// ------------ method that matches groomed and original jets based on minimum dR ------------
template <class IPTI, class VTX>
void TemplatedSecondaryVertexProducer<IPTI, VTX>::matchGroomedJets(const edm::Handle<edm::View<reco::Jet> > &jets,
                                                                   const edm::Handle<edm::View<reco::Jet> > &groomedJets,
                                                                   std::vector<int> &matchedIndices) {
  std::vector<bool> jetLocks(jets->size(), false);
  std::vector<int> jetIndices;

  for (size_t gj = 0; gj < groomedJets->size(); ++gj) {
    double matchedDR2 = 1e9;
    int matchedIdx = -1;

    if (groomedJets->at(gj).pt() > 0.)  // skip pathological cases of groomed jets with Pt=0
    {
      for (size_t j = 0; j < jets->size(); ++j) {
        if (jetLocks.at(j))
          continue;  // skip jets that have already been matched

        double tempDR2 = Geom::deltaR2(
            jets->at(j).rapidity(), jets->at(j).phi(), groomedJets->at(gj).rapidity(), groomedJets->at(gj).phi());
        if (tempDR2 < matchedDR2) {
          matchedDR2 = tempDR2;
          matchedIdx = j;
        }
      }
    }

    if (matchedIdx >= 0) {
      if (matchedDR2 > rParam * rParam) {
        edm::LogWarning("MatchedJetsFarApart")
            << "Matched groomed jet " << gj << " and original jet " << matchedIdx
            << " are separated by dR=" << sqrt(matchedDR2) << " which is greater than the jet size R=" << rParam
            << ".\n"
            << "This is not expected so the matching of these two jets has been discarded. Please check that the two "
               "jet collections belong to each other.";
        matchedIdx = -1;
      } else
        jetLocks.at(matchedIdx) = true;
    }
    jetIndices.push_back(matchedIdx);
  }

  for (size_t j = 0; j < jets->size(); ++j) {
    std::vector<int>::iterator matchedIndex = std::find(jetIndices.begin(), jetIndices.end(), j);

    matchedIndices.push_back(matchedIndex != jetIndices.end() ? std::distance(jetIndices.begin(), matchedIndex) : -1);
  }
}

// ------------ method that matches subjets and original fat jets ------------
template <class IPTI, class VTX>
void TemplatedSecondaryVertexProducer<IPTI, VTX>::matchSubjets(const std::vector<int> &groomedIndices,
                                                               const edm::Handle<edm::View<reco::Jet> > &groomedJets,
                                                               const edm::Handle<std::vector<IPTI> > &subjets,
                                                               std::vector<std::vector<int> > &matchedIndices) {
  for (size_t g = 0; g < groomedIndices.size(); ++g) {
    std::vector<int> subjetIndices;

    if (groomedIndices.at(g) >= 0) {
      for (size_t s = 0; s < groomedJets->at(groomedIndices.at(g)).numberOfDaughters(); ++s) {
        const edm::Ptr<reco::Candidate> &subjet = groomedJets->at(groomedIndices.at(g)).daughterPtr(s);

        for (size_t sj = 0; sj < subjets->size(); ++sj) {
          const edm::RefToBase<reco::Jet> &subjetRef = subjets->at(sj).jet();
          if (subjet == edm::Ptr<reco::Candidate>(subjetRef.id(), subjetRef.get(), subjetRef.key())) {
            subjetIndices.push_back(sj);
            break;
          }
        }
      }

      if (subjetIndices.empty())
        edm::LogError("SubjetMatchingFailed") << "Matching subjets to original fat jets failed. Please check that the "
                                                 "groomed fat jet and subjet collections belong to each other.";

      matchedIndices.push_back(subjetIndices);
    } else
      matchedIndices.push_back(subjetIndices);
  }
}

// ------------ method that matches subjets and original fat jets ------------
template <class IPTI, class VTX>
void TemplatedSecondaryVertexProducer<IPTI, VTX>::matchSubjets(const edm::Handle<edm::View<reco::Jet> > &fatJets,
                                                               const edm::Handle<std::vector<IPTI> > &subjets,
                                                               std::vector<std::vector<int> > &matchedIndices) {
  for (size_t fj = 0; fj < fatJets->size(); ++fj) {
    std::vector<int> subjetIndices;
    size_t nSubjetCollections = 0;
    size_t nSubjets = 0;

    const pat::Jet *fatJet = dynamic_cast<const pat::Jet *>(fatJets->ptrAt(fj).get());

    if (!fatJet) {
      if (fj == 0)
        edm::LogError("WrongJetType")
            << "Wrong jet type for input fat jets. Please check that the input fat jets are of the pat::Jet type.";

      matchedIndices.push_back(subjetIndices);
      continue;
    } else {
      nSubjetCollections = fatJet->subjetCollectionNames().size();

      if (nSubjetCollections > 0) {
        for (size_t coll = 0; coll < nSubjetCollections; ++coll) {
          const pat::JetPtrCollection &fatJetSubjets = fatJet->subjets(coll);

          for (size_t fjsj = 0; fjsj < fatJetSubjets.size(); ++fjsj) {
            ++nSubjets;

            for (size_t sj = 0; sj < subjets->size(); ++sj) {
              const pat::Jet *subJet = dynamic_cast<const pat::Jet *>(subjets->at(sj).jet().get());

              if (!subJet) {
                if (fj == 0 && coll == 0 && fjsj == 0 && sj == 0)
                  edm::LogError("WrongJetType") << "Wrong jet type for input subjets. Please check that the input "
                                                   "subjets are of the pat::Jet type.";

                break;
              } else {
                if (subJet->originalObjectRef() == fatJetSubjets.at(fjsj)->originalObjectRef()) {
                  subjetIndices.push_back(sj);
                  break;
                }
              }
            }
          }
        }

        if (subjetIndices.empty() && nSubjets > 0)
          edm::LogError("SubjetMatchingFailed") << "Matching subjets to fat jets failed. Please check that the fat jet "
                                                   "and subjet collections belong to each other.";

        matchedIndices.push_back(subjetIndices);
      } else
        matchedIndices.push_back(subjetIndices);
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
template <class IPTI, class VTX>
void TemplatedSecondaryVertexProducer<IPTI, VTX>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("extSVDeltaRToJet", 0.3);
  desc.add<edm::InputTag>("beamSpotTag", edm::InputTag("offlineBeamSpot"));
  {
    edm::ParameterSetDescription vertexReco;
    vertexReco.add<double>("primcut", 1.8);
    vertexReco.add<double>("seccut", 6.0);
    vertexReco.add<std::string>("finder", "avr");
    vertexReco.addOptionalNode(edm::ParameterDescription<double>("minweight", 0.5, true) and
                                   edm::ParameterDescription<double>("weightthreshold", 0.001, true) and
                                   edm::ParameterDescription<bool>("smoothing", false, true),
                               true);
    vertexReco.addOptionalNode(
        edm::ParameterDescription<double>("maxFitChi2", 10.0, true) and
            edm::ParameterDescription<double>("mergeThreshold", 3.0, true) and
            edm::ParameterDescription<std::string>("fitType", "RefitGhostTrackWithVertices", true),
        true);
    desc.add<edm::ParameterSetDescription>("vertexReco", vertexReco);
  }
  {
    edm::ParameterSetDescription vertexSelection;
    vertexSelection.add<std::string>("sortCriterium", "dist3dError");
    desc.add<edm::ParameterSetDescription>("vertexSelection", vertexSelection);
  }
  desc.add<std::string>("constraint", "BeamSpot");
  desc.add<edm::InputTag>("trackIPTagInfos", edm::InputTag("impactParameterTagInfos"));
  {
    edm::ParameterSetDescription vertexCuts;
    vertexCuts.add<double>("distSig3dMax", 99999.9);
    vertexCuts.add<double>("fracPV", 0.65);
    vertexCuts.add<double>("distVal2dMax", 2.5);
    vertexCuts.add<bool>("useTrackWeights", true);
    vertexCuts.add<double>("maxDeltaRToJetAxis", 0.4);
    {
      edm::ParameterSetDescription v0Filter;
      v0Filter.add<double>("k0sMassWindow", 0.05);
      vertexCuts.add<edm::ParameterSetDescription>("v0Filter", v0Filter);
    }
    vertexCuts.add<double>("distSig2dMin", 3.0);
    vertexCuts.add<unsigned int>("multiplicityMin", 2);
    vertexCuts.add<double>("distVal2dMin", 0.01);
    vertexCuts.add<double>("distSig2dMax", 99999.9);
    vertexCuts.add<double>("distVal3dMax", 99999.9);
    vertexCuts.add<double>("minimumTrackWeight", 0.5);
    vertexCuts.add<double>("distVal3dMin", -99999.9);
    vertexCuts.add<double>("massMax", 6.5);
    vertexCuts.add<double>("distSig3dMin", -99999.9);
    desc.add<edm::ParameterSetDescription>("vertexCuts", vertexCuts);
  }
  desc.add<bool>("useExternalSV", false);
  desc.add<double>("minimumTrackWeight", 0.5);
  desc.add<bool>("usePVError", true);
  {
    edm::ParameterSetDescription trackSelection;
    trackSelection.add<double>("b_pT", 0.3684);
    trackSelection.add<double>("max_pT", 500);
    trackSelection.add<bool>("useVariableJTA", false);
    trackSelection.add<double>("maxDecayLen", 99999.9);
    trackSelection.add<double>("sip3dValMin", -99999.9);
    trackSelection.add<double>("max_pT_dRcut", 0.1);
    trackSelection.add<double>("a_pT", 0.005263);
    trackSelection.add<unsigned int>("totalHitsMin", 8);
    trackSelection.add<double>("jetDeltaRMax", 0.3);
    trackSelection.add<double>("a_dR", -0.001053);
    trackSelection.add<double>("maxDistToAxis", 0.2);
    trackSelection.add<double>("ptMin", 1.0);
    trackSelection.add<std::string>("qualityClass", "any");
    trackSelection.add<unsigned int>("pixelHitsMin", 2);
    trackSelection.add<double>("sip2dValMax", 99999.9);
    trackSelection.add<double>("max_pT_trackPTcut", 3);
    trackSelection.add<double>("sip2dValMin", -99999.9);
    trackSelection.add<double>("normChi2Max", 99999.9);
    trackSelection.add<double>("sip3dValMax", 99999.9);
    trackSelection.add<double>("sip3dSigMin", -99999.9);
    trackSelection.add<double>("min_pT", 120);
    trackSelection.add<double>("min_pT_dRcut", 0.5);
    trackSelection.add<double>("sip2dSigMax", 99999.9);
    trackSelection.add<double>("sip3dSigMax", 99999.9);
    trackSelection.add<double>("sip2dSigMin", -99999.9);
    trackSelection.add<double>("b_dR", 0.6263);
    desc.add<edm::ParameterSetDescription>("trackSelection", trackSelection);
  }
  desc.add<std::string>("trackSort", "sip3dSig");
  desc.add<edm::InputTag>("extSVCollection", edm::InputTag("secondaryVertices"));
  desc.addOptionalNode(edm::ParameterDescription<bool>("useSVClustering", false, true) and
                           edm::ParameterDescription<std::string>("jetAlgorithm", true) and
                           edm::ParameterDescription<double>("rParam", true),
                       true);
  desc.addOptional<bool>("useSVMomentum", false);
  desc.addOptional<double>("ghostRescaling", 1e-18);
  desc.addOptional<double>("relPtTolerance", 1e-03);
  desc.addOptional<edm::InputTag>("fatJets");
  desc.addOptional<edm::InputTag>("groomedFatJets");
  desc.add<edm::InputTag>("weights", edm::InputTag(""));
  descriptions.addDefault(desc);
}

//define this as a plug-in
typedef TemplatedSecondaryVertexProducer<TrackIPTagInfo, reco::Vertex> SecondaryVertexProducer;
typedef TemplatedSecondaryVertexProducer<CandIPTagInfo, reco::VertexCompositePtrCandidate> CandSecondaryVertexProducer;

DEFINE_FWK_MODULE(SecondaryVertexProducer);
DEFINE_FWK_MODULE(CandSecondaryVertexProducer);

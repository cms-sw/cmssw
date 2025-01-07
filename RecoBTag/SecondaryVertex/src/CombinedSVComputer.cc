#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"

using namespace reco;

inline edm::ParameterSet CombinedSVComputer::dropDeltaR(const edm::ParameterSet &pset) const {
  edm::ParameterSet psetCopy(pset);
  psetCopy.addParameter<double>("jetDeltaRMax", 99999.0);
  return psetCopy;
}

CombinedSVComputer::CombinedSVComputer(const edm::ParameterSet &params)
    : trackFlip(params.getParameter<bool>("trackFlip")),
      vertexFlip(params.getParameter<bool>("vertexFlip")),
      charmCut(params.getParameter<double>("charmCut")),
      sortCriterium(TrackSorting::getCriterium(params.getParameter<std::string>("trackSort"))),
      trackSelector(params.getParameter<edm::ParameterSet>("trackSelection")),
      trackNoDeltaRSelector(dropDeltaR(params.getParameter<edm::ParameterSet>("trackSelection"))),
      trackPseudoSelector(params.getParameter<edm::ParameterSet>("trackPseudoSelection")),
      pseudoMultiplicityMin(params.getParameter<unsigned int>("pseudoMultiplicityMin")),
      trackMultiplicityMin(params.getParameter<unsigned int>("trackMultiplicityMin")),
      minTrackWeight(params.getParameter<double>("minimumTrackWeight")),
      useTrackWeights(params.getParameter<bool>("useTrackWeights")),
      vertexMassCorrection(params.getParameter<bool>("correctVertexMass")),
      pseudoVertexV0Filter(params.getParameter<edm::ParameterSet>("pseudoVertexV0Filter")),
      trackPairV0Filter(params.getParameter<edm::ParameterSet>("trackPairV0Filter")) {}

inline double CombinedSVComputer::flipValue(double value, bool vertex) const {
  return (vertex ? vertexFlip : trackFlip) ? -value : value;
}

inline CombinedSVComputer::IterationRange CombinedSVComputer::flipIterate(int size, bool vertex) const {
  IterationRange range;
  if (vertex ? vertexFlip : trackFlip) {
    range.begin = size - 1;
    range.end = -1;
    range.increment = -1;
  } else {
    range.begin = 0;
    range.end = size;
    range.increment = +1;
  }

  return range;
}

const btag::TrackIPData &CombinedSVComputer::threshTrack(const CandIPTagInfo &trackIPTagInfo,
                                                         const btag::SortCriteria sort,
                                                         const reco::Jet &jet,
                                                         const GlobalPoint &pv) const {
  const CandIPTagInfo::input_container &tracks = trackIPTagInfo.selectedTracks();
  const std::vector<btag::TrackIPData> &ipData = trackIPTagInfo.impactParameterData();
  std::vector<std::size_t> indices = trackIPTagInfo.sortedIndexes(sort);

  IterationRange range = flipIterate(indices.size(), false);
  TrackKinematics kin;
  range_for(i, range) {
    std::size_t idx = indices[i];
    const btag::TrackIPData &data = ipData[idx];
    const CandidatePtr &track = tracks[idx];

    if (!trackNoDeltaRSelector(track, data, jet, pv))
      continue;

    kin.add(track);
    if (kin.vectorSum().M() > charmCut)
      return data;
  }
  if (trackFlip) {
    static const btag::TrackIPData dummy = {GlobalPoint(),
                                            GlobalPoint(),
                                            Measurement1D(1.0, 1.0),
                                            Measurement1D(1.0, 1.0),
                                            Measurement1D(1.0, 1.0),
                                            Measurement1D(1.0, 1.0),
                                            0.};
    return dummy;
  } else {
    static const btag::TrackIPData dummy = {GlobalPoint(),
                                            GlobalPoint(),
                                            Measurement1D(-1.0, 1.0),
                                            Measurement1D(-1.0, 1.0),
                                            Measurement1D(-1.0, 1.0),
                                            Measurement1D(-1.0, 1.0),
                                            0.};
    return dummy;
  }
}

const btag::TrackIPData &CombinedSVComputer::threshTrack(const TrackIPTagInfo &trackIPTagInfo,
                                                         const btag::SortCriteria sort,
                                                         const reco::Jet &jet,
                                                         const GlobalPoint &pv) const {
  const edm::RefVector<TrackCollection> &tracks = trackIPTagInfo.selectedTracks();
  const std::vector<btag::TrackIPData> &ipData = trackIPTagInfo.impactParameterData();
  std::vector<std::size_t> indices = trackIPTagInfo.sortedIndexes(sort);

  IterationRange range = flipIterate(indices.size(), false);
  TrackKinematics kin;
  range_for(i, range) {
    std::size_t idx = indices[i];
    const btag::TrackIPData &data = ipData[idx];
    const Track &track = *tracks[idx];

    if (!trackNoDeltaRSelector(track, data, jet, pv))
      continue;

    kin.add(track);
    if (kin.vectorSum().M() > charmCut)
      return data;
  }

  if (trackFlip) {
    static const btag::TrackIPData dummy = {GlobalPoint(),
                                            GlobalPoint(),
                                            Measurement1D(1.0, 1.0),
                                            Measurement1D(1.0, 1.0),
                                            Measurement1D(1.0, 1.0),
                                            Measurement1D(1.0, 1.0),
                                            0.};
    return dummy;
  } else {
    static const btag::TrackIPData dummy = {GlobalPoint(),
                                            GlobalPoint(),
                                            Measurement1D(-1.0, 1.0),
                                            Measurement1D(-1.0, 1.0),
                                            Measurement1D(-1.0, 1.0),
                                            Measurement1D(-1.0, 1.0),
                                            0.};
    return dummy;
  }
}

TaggingVariableList CombinedSVComputer::operator()(const TrackIPTagInfo &ipInfo,
                                                   const SecondaryVertexTagInfo &svInfo) const {
  using namespace ROOT::Math;

  edm::RefToBase<Jet> jet = ipInfo.jet();
  math::XYZVector jetDir = jet->momentum().Unit();
  TaggingVariableList vars;

  TrackKinematics vertexKinematics;

  double vtx_track_ptSum = 0.;
  double vtx_track_ESum = 0.;

  // the following is specific depending on the type of vertex
  int vtx = -1;
  unsigned int numberofvertextracks = 0;

  IterationRange range = flipIterate(svInfo.nVertices(), true);
  range_for(i, range) {
    numberofvertextracks = numberofvertextracks + (svInfo.secondaryVertex(i)).nTracks();

    const Vertex &vertex = svInfo.secondaryVertex(i);
    bool hasRefittedTracks = vertex.hasRefittedTracks();
    for (reco::Vertex::trackRef_iterator track = vertex.tracks_begin(); track != vertex.tracks_end(); track++) {
      double w = vertex.trackWeight(*track);
      if (w < minTrackWeight)
        continue;
      if (hasRefittedTracks) {
        const Track actualTrack = vertex.refittedTrack(*track);
        vertexKinematics.add(actualTrack, w);
        vars.insert(btau::trackEtaRel, reco::btau::etaRel(jetDir, actualTrack.momentum()), true);
        if (vtx < 0)  // calculate this only for the first vertex
        {
          vtx_track_ptSum += std::sqrt(actualTrack.momentum().Perp2());
          vtx_track_ESum += std::sqrt(actualTrack.momentum().Mag2() + ROOT::Math::Square(ParticleMasses::piPlus));
        }
      } else {
        vertexKinematics.add(**track, w);
        vars.insert(btau::trackEtaRel, reco::btau::etaRel(jetDir, (*track)->momentum()), true);
        if (vtx < 0)  // calculate this only for the first vertex
        {
          vtx_track_ptSum += std::sqrt((*track)->momentum().Perp2());
          vtx_track_ESum += std::sqrt((*track)->momentum().Mag2() + ROOT::Math::Square(ParticleMasses::piPlus));
        }
      }
    }

    if (vtx < 0)
      vtx = i;
  }
  if (vtx >= 0) {
    vars.insert(btau::vertexNTracks, numberofvertextracks, true);
    vars.insert(btau::vertexFitProb, (svInfo.secondaryVertex(vtx)).normalizedChi2(), true);
  }

  // after we collected vertex information we let the common code complete the job
  fillCommonVariables(vars, vertexKinematics, ipInfo, svInfo, vtx_track_ptSum, vtx_track_ESum);

  vars.finalize();
  return vars;
}

TaggingVariableList CombinedSVComputer::operator()(const CandIPTagInfo &ipInfo,
                                                   const CandSecondaryVertexTagInfo &svInfo) const {
  using namespace ROOT::Math;

  edm::RefToBase<Jet> jet = ipInfo.jet();
  math::XYZVector jetDir = jet->momentum().Unit();
  TaggingVariableList vars;

  TrackKinematics vertexKinematics;

  double vtx_track_ptSum = 0.;
  double vtx_track_ESum = 0.;

  // the following is specific depending on the type of vertex
  int vtx = -1;
  unsigned int numberofvertextracks = 0;

  IterationRange range = flipIterate(svInfo.nVertices(), true);
  range_for(i, range) {
    numberofvertextracks = numberofvertextracks + (svInfo.secondaryVertex(i)).numberOfSourceCandidatePtrs();

    const reco::VertexCompositePtrCandidate &vertex = svInfo.secondaryVertex(i);
    const std::vector<CandidatePtr> &tracks = vertex.daughterPtrVector();
    for (std::vector<CandidatePtr>::const_iterator track = tracks.begin(); track != tracks.end(); ++track) {
      vertexKinematics.add(*track);
      vars.insert(btau::trackEtaRel, reco::btau::etaRel(jetDir, (*track)->momentum()), true);
      if (vtx < 0)  // calculate this only for the first vertex
      {
        vtx_track_ptSum += std::sqrt((*track)->momentum().Perp2());
        vtx_track_ESum += std::sqrt((*track)->momentum().Mag2() + ROOT::Math::Square(ParticleMasses::piPlus));
      }
    }

    if (vtx < 0)
      vtx = i;
  }
  if (vtx >= 0) {
    vars.insert(btau::vertexNTracks, numberofvertextracks, true);
    vars.insert(btau::vertexFitProb, (svInfo.secondaryVertex(vtx)).vertexNormalizedChi2(), true);
  }

  // after we collected vertex information we let the common code complete the job
  fillCommonVariables(vars, vertexKinematics, ipInfo, svInfo, vtx_track_ptSum, vtx_track_ESum);

  vars.finalize();
  return vars;
}

void CombinedSVComputer::fillPSetDescription(edm::ParameterSetDescription &desc) {
  {
    edm::ParameterSetDescription trackPseudoSelection;
    trackPseudoSelection.add<double>("max_pT_dRcut", 0.1);
    trackPseudoSelection.add<double>("b_dR", 0.6263);
    trackPseudoSelection.add<double>("min_pT", 120.0);
    trackPseudoSelection.add<double>("b_pT", 0.3684);
    trackPseudoSelection.add<double>("ptMin", 0.0);
    trackPseudoSelection.add<double>("max_pT_trackPTcut", 3.0);
    trackPseudoSelection.add<double>("max_pT", 500.0);
    trackPseudoSelection.add<bool>("useVariableJTA", false);
    trackPseudoSelection.add<double>("maxDecayLen", 5.0);
    trackPseudoSelection.add<std::string>("qualityClass", "any");
    trackPseudoSelection.add<double>("normChi2Max", 99999.9);
    trackPseudoSelection.add<double>("sip2dValMin", -99999.9);
    trackPseudoSelection.add<double>("sip3dValMin", -99999.9);
    trackPseudoSelection.add<double>("a_dR", -0.001053);
    trackPseudoSelection.add<double>("maxDistToAxis", 0.07);
    trackPseudoSelection.add<uint32_t>("totalHitsMin", 3);
    trackPseudoSelection.add<double>("a_pT", 0.005263);
    trackPseudoSelection.add<double>("sip2dSigMax", 99999.9);
    trackPseudoSelection.add<double>("sip2dValMax", 99999.9);
    trackPseudoSelection.add<double>("sip3dSigMax", 99999.9);
    trackPseudoSelection.add<double>("sip3dValMax", 99999.9);
    trackPseudoSelection.add<double>("min_pT_dRcut", 0.5);
    trackPseudoSelection.add<double>("jetDeltaRMax", 0.3);
    trackPseudoSelection.add<uint32_t>("pixelHitsMin", 0);
    trackPseudoSelection.add<double>("sip3dSigMin", -99999.9);
    trackPseudoSelection.add<double>("sip2dSigMin", 2.0);
    desc.add("trackPseudoSelection", trackPseudoSelection);
  }

  {
    edm::ParameterSetDescription trackSelection;
    trackSelection.add<double>("max_pT_dRcut", 0.1);
    trackSelection.add<double>("b_dR", 0.6263);
    trackSelection.add<double>("min_pT", 120.0);
    trackSelection.add<double>("b_pT", 0.3684);
    trackSelection.add<double>("ptMin", 0.0);
    trackSelection.add<double>("max_pT_trackPTcut", 3.0);
    trackSelection.add<double>("max_pT", 500.0);
    trackSelection.add<bool>("useVariableJTA", false);
    trackSelection.add<double>("maxDecayLen", 5.0);
    trackSelection.add<std::string>("qualityClass", "any");
    trackSelection.add<double>("normChi2Max", 99999.9);
    trackSelection.add<double>("sip2dValMin", -99999.9);
    trackSelection.add<double>("sip3dValMin", -99999.9);
    trackSelection.add<double>("a_dR", -0.001053);
    trackSelection.add<double>("maxDistToAxis", 0.07);
    trackSelection.add<uint32_t>("totalHitsMin", 3);
    trackSelection.add<double>("a_pT", 0.005263);
    trackSelection.add<double>("sip2dSigMax", 99999.9);
    trackSelection.add<double>("sip2dValMax", 99999.9);
    trackSelection.add<double>("sip3dSigMax", 99999.9);
    trackSelection.add<double>("sip3dValMax", 99999.9);
    trackSelection.add<double>("min_pT_dRcut", 0.5);
    trackSelection.add<double>("jetDeltaRMax", 0.3);
    trackSelection.add<uint32_t>("pixelHitsMin", 2);
    trackSelection.add<double>("sip3dSigMin", -99999.9);
    trackSelection.add<double>("sip2dSigMin", -99999.9);
    desc.add("trackSelection", trackSelection);
  }

  edm::ParameterSetDescription trackPairV0Filter;
  trackPairV0Filter.add<double>("k0sMassWindow", 0.03);
  desc.add("trackPairV0Filter", trackPairV0Filter);

  edm::ParameterSetDescription pseudoVertexV0Filter;
  pseudoVertexV0Filter.add<double>("k0sMassWindow", 0.05);
  desc.add("pseudoVertexV0Filter", pseudoVertexV0Filter);

  desc.add<bool>("trackFlip", false);
  desc.add<bool>("useTrackWeights", true);
  desc.add<bool>("SoftLeptonFlip", false);
  desc.add<uint32_t>("pseudoMultiplicityMin", 2);
  desc.add<bool>("correctVertexMass", true);
  desc.add<double>("minimumTrackWeight", 0.5);
  desc.add<double>("charmCut", 1.5);
  desc.add<std::string>("trackSort", "sip2dSig");
  desc.add<uint32_t>("trackMultiplicityMin", 2);
  desc.add<bool>("vertexFlip", false);
}

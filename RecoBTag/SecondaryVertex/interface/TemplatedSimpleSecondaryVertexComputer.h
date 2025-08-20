#ifndef RecoBTag_SecondaryVertex_TemplatedSimpleSecondaryVertexComputer_h
#define RecoBTag_SecondaryVertex_TemplatedSimpleSecondaryVertexComputer_h

#include <cmath>

#include "DataFormats/BTauReco/interface/TemplatedSecondaryVertexTagInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "Math/GenVector/VectorUtil.h"

template <class IPTI, class VTX>
class TemplatedSimpleSecondaryVertexComputer : public JetTagComputer {
public:
  using Tokens = void;

  typedef reco::TemplatedSecondaryVertexTagInfo<IPTI, VTX> TagInfo;

  TemplatedSimpleSecondaryVertexComputer(const edm::ParameterSet &parameters)
      : use2d(!parameters.getParameter<bool>("use3d")),
        useSig(parameters.getParameter<bool>("useSignificance")),
        unBoost(parameters.getParameter<bool>("unBoost")),
        minTracks(parameters.getParameter<unsigned int>("minTracks")),
        minVertices_(parameters.getParameter<unsigned int>("minVertices")) {
    uses("svTagInfos");
  }

  static void fillPSetDescription(edm::ParameterSetDescription &desc) {
    desc.add<bool>("use3d", true);
    desc.add<bool>("useSignificance", true);
    desc.add<bool>("unBoost", false);
    desc.add<unsigned int>("minTracks", 2);
    desc.add<unsigned int>("minVertices", 1);
  }

  float discriminator(const TagInfoHelper &tagInfos) const override {
    const TagInfo &info = tagInfos.get<TagInfo>();
    if (info.nVertices() < minVertices_)
      return -1;
    unsigned int idx = 0;
    while (idx < info.nVertices()) {
      if (info.nVertexTracks(idx) >= minTracks)
        break;
      idx++;
    }
    if (idx >= info.nVertices())
      return -1.0;

    double gamma;
    if (unBoost) {
      reco::TrackKinematics kinematics(info.secondaryVertex(idx));
      gamma = kinematics.vectorSum().Gamma();
    } else
      gamma = 1.0;

    double value;
    if (useSig)
      value = info.flightDistance(idx, use2d).significance();
    else
      value = info.flightDistance(idx, use2d).value();

    value /= gamma;

    if (useSig)
      value = (value > 0) ? +std::log(1 + value) : -std::log(1 - value);

    return value;
  }

private:
  bool use2d;
  bool useSig;
  bool unBoost;
  unsigned int minTracks;
  unsigned int minVertices_;
};

#endif  // RecoBTag_SecondaryVertex_TemplatedSimpleSecondaryVertexComputer_h

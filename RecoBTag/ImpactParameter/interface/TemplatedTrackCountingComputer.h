#ifndef ImpactParameter_TemplatedTrackCountingComputer_h
#define ImpactParameter_TemplatedTrackCountingComputer_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/IPTagInfo.h"
#include "Math/GenVector/VectorUtil.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

template <class Container, class Base>
class TemplatedTrackCountingComputer : public JetTagComputer {
public:
  using Tokens = void;

  typedef reco::IPTagInfo<Container, Base> TagInfo;

  TemplatedTrackCountingComputer(const edm::ParameterSet& parameters) {
    m_minIP = parameters.existsAs<double>("minimumImpactParameter")
                  ? parameters.getParameter<double>("minimumImpactParameter")
                  : -1;
    m_useSignedIPSig = parameters.existsAs<bool>("useSignedImpactParameterSig")
                           ? parameters.getParameter<bool>("useSignedImpactParameterSig")
                           : true;
    m_nthTrack = parameters.getParameter<int>("nthTrack");
    m_ipType = parameters.getParameter<int>("impactParameterType");
    m_deltaR = parameters.getParameter<double>("deltaR");
    m_cutMaxDecayLen = parameters.getParameter<double>("maximumDecayLength");          //used
    m_cutMaxDistToAxis = parameters.getParameter<double>("maximumDistanceToJetAxis");  //used
    //
    // access track quality class; "any" takes everything
    //
    std::string trackQualityType = parameters.getParameter<std::string>("trackQualityClass");  //used
    m_trackQuality = reco::TrackBase::qualityByName(trackQualityType);
    m_useAllQualities = false;
    if (trackQualityType == "any" || trackQualityType == "Any" || trackQualityType == "ANY")
      m_useAllQualities = true;

    uses("ipTagInfos");

    useVariableJTA_ =
        parameters.existsAs<bool>("useVariableJTA") ? parameters.getParameter<bool>("useVariableJTA") : false;
    if (useVariableJTA_) {
      varJTApars = {parameters.getParameter<double>("a_dR"),
                    parameters.getParameter<double>("b_dR"),
                    parameters.getParameter<double>("a_pT"),
                    parameters.getParameter<double>("b_pT"),
                    parameters.getParameter<double>("min_pT"),
                    parameters.getParameter<double>("max_pT"),
                    parameters.getParameter<double>("min_pT_dRcut"),
                    parameters.getParameter<double>("max_pT_dRcut"),
                    parameters.getParameter<double>("max_pT_trackPTcut")};
    }
  }

  float discriminator(const TagInfoHelper& ti) const override {
    const TagInfo& tkip = ti.get<TagInfo>();
    std::multiset<float> significances = orderedSignificances(tkip);
    std::multiset<float>::reverse_iterator nth = significances.rbegin();
    for (int i = 0; i < m_nthTrack - 1 && nth != significances.rend(); i++)
      nth++;
    if (nth != significances.rend())
      return *nth;
    else
      return -100.;
  }

protected:
  std::multiset<float> orderedSignificances(const TagInfo& tkip) const {
    const std::vector<reco::btag::TrackIPData>& impactParameters((tkip.impactParameterData()));
    const Container& tracks(tkip.selectedTracks());
    std::multiset<float> significances;
    int i = 0;
    if (tkip.primaryVertex().isNull()) {
      return std::multiset<float>();
    }

    GlobalPoint pv(tkip.primaryVertex()->position().x(),
                   tkip.primaryVertex()->position().y(),
                   tkip.primaryVertex()->position().z());

    for (std::vector<reco::btag::TrackIPData>::const_iterator it = impactParameters.begin();
         it != impactParameters.end();
         ++it, i++) {
      if (fabs(impactParameters[i].distanceToJetAxis.value()) < m_cutMaxDistToAxis &&  // distance to JetAxis
          (impactParameters[i].closestToJetAxis - pv).mag() < m_cutMaxDecayLen &&      // max decay len
          (m_useAllQualities == true ||
           reco::btag::toTrack(tracks[i])->quality(m_trackQuality)) &&       // use selected track qualities
          (fabs(((m_ipType == 0) ? it->ip3d : it->ip2d).value()) > m_minIP)  // minimum impact parameter
      ) {
        //calculate the signed or un-signed significance
        float signed_sig = ((m_ipType == 0) ? it->ip3d : it->ip2d).significance();
        float unsigned_sig = fabs(signed_sig);
        float significance = (m_useSignedIPSig) ? signed_sig : unsigned_sig;

        if (useVariableJTA_) {
          if (tkip.variableJTA(varJTApars)[i])
            significances.insert(significance);
        } else  // no using variable JTA, use the default method
          if (m_deltaR <= 0 ||
              ROOT::Math::VectorUtil::DeltaR((*tkip.jet()).p4().Vect(), (*tracks[i]).momentum()) < m_deltaR)
            significances.insert(significance);
      }
    }

    return significances;
  }

  bool useVariableJTA_;
  reco::btag::variableJTAParameters varJTApars;

  double m_minIP;
  bool m_useSignedIPSig;

  int m_nthTrack;
  int m_ipType;
  double m_deltaR;
  double m_cutMaxDecayLen;
  double m_cutMaxDistToAxis;
  reco::TrackBase::TrackQuality m_trackQuality;
  bool m_useAllQualities;
};

#endif  // ImpactParameter_TemplatedTrackCountingComputer_h

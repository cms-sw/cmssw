#ifndef RecoBTag_SecondaryVertex_CombinedSVSoftLeptonComputer_h
#define RecoBTag_SecondaryVertex_CombinedSVSoftLeptonComputer_h

#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/TemplatedSoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/VertexTypes.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"

class CombinedSVSoftLeptonComputer : public CombinedSVComputer {
public:
  explicit CombinedSVSoftLeptonComputer(const edm::ParameterSet &params);
  ~CombinedSVSoftLeptonComputer() override = default;
  double flipSoftLeptonValue(double value) const;

  template <class IPTI, class SVTI>
  reco::TaggingVariableList operator()(const IPTI &ipInfo,
                                       const SVTI &svInfo,
                                       const reco::CandSoftLeptonTagInfo &muonInfo,
                                       const reco::CandSoftLeptonTagInfo &elecInfo) const;

private:
  bool SoftLeptonFlip;
};

inline double CombinedSVSoftLeptonComputer::flipSoftLeptonValue(double value) const {
  return SoftLeptonFlip ? -value : value;
}

template <class IPTI, class SVTI>
reco::TaggingVariableList CombinedSVSoftLeptonComputer::operator()(const IPTI &ipInfo,
                                                                   const SVTI &svInfo,
                                                                   const reco::CandSoftLeptonTagInfo &muonInfo,
                                                                   const reco::CandSoftLeptonTagInfo &elecInfo) const {
  using namespace reco;

  // call the inherited operator()
  TaggingVariableList vars = CombinedSVComputer::operator()(ipInfo, svInfo);

  //Jets with vtxCategory 99 cause problems
  unsigned int vtxType =
      (vars.checkTag(reco::btau::vertexCategory) ? (unsigned int)(vars.get(reco::btau::vertexCategory)) : 99);
  if (vtxType == 99)
    return vars;

  // the following is specific to soft leptons
  int leptonCategory = 0;  // 0 = no lepton, 1 = muon, 2 = electron

  for (unsigned int i = 0; i < muonInfo.leptons();
       ++i)  // loop over all muons, not optimal -> find the best or use ranking from best to worst
  {
    leptonCategory = 1;  // muon category
    const SoftLeptonProperties &propertiesMuon = muonInfo.properties(i);
    vars.insert(btau::leptonPtRel, propertiesMuon.ptRel, true);
    vars.insert(btau::leptonSip3d, flipSoftLeptonValue(propertiesMuon.sip3d), true);
    vars.insert(btau::leptonDeltaR, propertiesMuon.deltaR, true);
    vars.insert(btau::leptonRatioRel, propertiesMuon.ratioRel, true);
    vars.insert(btau::leptonEtaRel, propertiesMuon.etaRel, true);
    vars.insert(btau::leptonRatio, propertiesMuon.ratio, true);
  }

  if (leptonCategory != 1)  // no soft muon found, try soft electron
  {
    for (unsigned int i = 0; i < elecInfo.leptons();
         ++i)  // loop over all electrons, not optimal -> find the best or use ranking from best to worst
    {
      leptonCategory = 2;  // electron category
      const SoftLeptonProperties &propertiesElec = elecInfo.properties(i);
      vars.insert(btau::leptonPtRel, propertiesElec.ptRel, true);
      vars.insert(btau::leptonSip3d, flipSoftLeptonValue(propertiesElec.sip3d), true);
      vars.insert(btau::leptonDeltaR, propertiesElec.deltaR, true);
      vars.insert(btau::leptonRatioRel, propertiesElec.ratioRel, true);
      vars.insert(btau::leptonEtaRel, propertiesElec.etaRel, true);
      vars.insert(btau::leptonRatio, propertiesElec.ratio, true);
    }
  }

  // set the default value for vertexLeptonCategory to 2 (= NoVertexNoSoftLepton)
  int vertexLepCat = 2;

  if (leptonCategory == 0)  // no soft lepton
  {
    if (vtxType == (unsigned int)(btag::Vertices::RecoVertex))
      vertexLepCat = 0;
    else if (vtxType == (unsigned int)(btag::Vertices::PseudoVertex))
      vertexLepCat = 1;
    else
      vertexLepCat = 2;
  } else if (leptonCategory == 1)  // soft muon
  {
    if (vtxType == (unsigned int)(btag::Vertices::RecoVertex))
      vertexLepCat = 3;
    else if (vtxType == (unsigned int)(btag::Vertices::PseudoVertex))
      vertexLepCat = 4;
    else
      vertexLepCat = 5;
  } else if (leptonCategory == 2)  // soft electron
  {
    if (vtxType == (unsigned int)(btag::Vertices::RecoVertex))
      vertexLepCat = 6;
    else if (vtxType == (unsigned int)(btag::Vertices::PseudoVertex))
      vertexLepCat = 7;
    else
      vertexLepCat = 8;
  }
  vars.insert(btau::vertexLeptonCategory, vertexLepCat, true);

  vars.finalize();
  return vars;
}

#endif  // RecoBTag_SecondaryVertex_CombinedSVSoftLeptonComputer_h

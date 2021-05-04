#ifndef RecoJets_JetAlgorithms_interface_CATopJetHelper_h
#define RecoJets_JetAlgorithms_interface_CATopJetHelper_h

// \class CATopJetHelper
//
// \short Create tag info properties for CATopTags that can be computed
//        "on the fly".
//
//
// \author Salvatore Rappoccio
// \version first version on 1-May-2011

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/CATopJetTagInfo.h"

class CATopJetHelper {
public:
  CATopJetHelper(double TopMass, double WMass) : TopMass_(TopMass), WMass_(WMass) {}

  reco::CATopJetProperties operator()(reco::Jet const& ihardJet) const;

protected:
  double TopMass_;
  double WMass_;
};

#endif

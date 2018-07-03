#ifndef DataFormats_BTauReco_interface_HTTTopJetTagInfo_h
#define DataFormats_BTauReco_interface_HTTTopJetTagInfo_h


// \class HTTTopJetTagInfo
// 
// \short specific tag info for HEPTopTagger tagging algorithm
// HTTTopJetTagInfo is a class to hold the discriminator variables for the
// HEPTopTagger algorithm.
// 
//
// \author Gregor Kasieczka (based on  CATopJetTagInfo by Salvatore Rappoccio)
// \version first version on 25 Sep 2014

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include <vector>

namespace reco {
 
class HTTTopJetProperties {
public:
  HTTTopJetProperties() : fjPt(0.),
    fjMass(0.),
    fjEta(0.),
    fjPhi(0.),
    topMass(0.),
    unfilteredMass(0.),
    prunedMass(0.),
    fRec(0.),
    massRatioPassed(0.),
    ropt(0.),
    roptCalc(0.),
    ptForRoptCalc(0.),
    tau1Unfiltered(0.),
    tau2Unfiltered(0.),
    tau3Unfiltered(0.),
    tau1Filtered(0.),
    tau2Filtered(0.),
    tau3Filtered(0.),
    qWeight(0.),
    qEpsilon(0.),
    qSigmaM(0.) {}

  double              fjPt;             //<! Mass of the inital Fatjet passed to the TT
  double              fjMass;           //<! Mass of the inital Fatjet passed to the TT
  double              fjEta;            //<! Mass of the inital Fatjet passed to the TT
  double              fjPhi;            //<! Mass of the inital Fatjet passed to the TT
  double              topMass;          //<! Mass of the HTT top quark candidate [GeV]
  double              unfilteredMass;   //<! Unfiltered mass of the triplet [GeV]
  double              prunedMass;       //<! Mass of the pruned fat jet [GeV]
  double              fRec;             //<! Minimum distance of m_ij/m_123 from m_W/m_top
  double              massRatioPassed;  //<! Did the candidate pass the default mass ratio? 
  double              ropt;             //<! R_opt found in Optimal procedure. 
  double              roptCalc;         //<! R_opt calc for a top quark based on filtered fat-jet pT.
  double              ptForRoptCalc;    //<! Filtered initial fatjet pT calculation of Ropt 
  double              tau1Unfiltered;   //<! 1-subjettiness, no filtering
  double              tau2Unfiltered;   //<! 2-subjettiness, no filtering
  double              tau3Unfiltered;   //<! 3-subjettiness, no filtering
  double              tau1Filtered;     //<! 1-subjettiness, with filtering
  double              tau2Filtered;     //<! 2-subjettiness, with filtering
  double              tau3Filtered;     //<! 3-subjettiness, with filtering
  double              qWeight;          //<! maximal weight of jet using Q-jet approach
  double              qEpsilon;         //<! fraction of jets tagged with Q-jets
  double              qSigmaM;          //<! Width of Q-jet mass distribution
};

 class HTTTopJetTagInfo : public JetTagInfo {
public:
  typedef edm::RefToBase<Jet> jet_type;
  typedef HTTTopJetProperties  properties_type;
    
    HTTTopJetTagInfo(void) {}

    ~HTTTopJetTagInfo(void) override {}
  
    HTTTopJetTagInfo* clone(void) const override { return new HTTTopJetTagInfo(*this); }
    
    const properties_type & properties() const {
      return properties_;
    }

    void insert(const edm::RefToBase<Jet> & jet, const HTTTopJetProperties & properties) {
      setJetRef(jet);
      properties_ = properties;
    }

protected:
    properties_type properties_;

};

DECLARE_EDM_REFS( HTTTopJetTagInfo )

}

#endif // AnalysisDataFormats_TopObjects_interface_HTTTopJetTagInfo_h

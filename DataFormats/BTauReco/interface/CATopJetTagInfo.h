#ifndef AnalysisDataFormats_TopObjects_interface_CATopJetTagInfo_h
#define AnalysisDataFormats_TopObjects_interface_CATopJetTagInfo_h


// \class CATopJetTagInfo
// 
// \short tag info for Cambridge-Aachen based subjet top-jet tagging algorithm
// CATopJetTagInfo is a class to hold the discriminator variables for the
// CATopJet Tagging algorithm.
// 
//
// \author Salvatore Rappoccio
// \version first version on 27 Aug 2008

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include <vector>

namespace reco {
 
class CATopJetProperties {
public:
  CATopJetProperties() {
    nSubJets = 0;
    minMass = 0.;
    topMass = 0.;
    wMass = 0.;
  }
  int                 nSubJets;        //<! Number of subjets
  double              minMass;         //<! Minimum invariant mass pairing
  double              topMass;         //<! Jet mass
  double              wMass;           //<! Closest mass to W mass
};

 class CATopJetTagInfo : public JetTagInfo {
public:
  typedef edm::RefToBase<Jet> jet_type;
  typedef CATopJetProperties  properties_type;
    
    CATopJetTagInfo(void) {}

    virtual ~CATopJetTagInfo(void) {}
  
    virtual CATopJetTagInfo* clone(void) const { return new CATopJetTagInfo(*this); }
    
    const properties_type & properties() const {
      return properties_;
    }

    void insert(const edm::RefToBase<Jet> & jet, const CATopJetProperties & properties) {
      setJetRef(jet);
      properties_ = properties;
    }

protected:
    properties_type properties_;

};

DECLARE_EDM_REFS( CATopJetTagInfo )

}

#endif // AnalysisDataFormats_TopObjects_interface_CATopJetTagInfo_h

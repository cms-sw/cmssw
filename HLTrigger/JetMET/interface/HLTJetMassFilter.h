#ifndef HLTJetMassFilter_h
#define HLTJetMassFilter_h

//system includ files
#include <memory>
#include <vector>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/JetReco/interface/CATopJetTagInfo.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <Math/VectorUtil.h>


//class declaration

class HLTJetMassFilter : public HLTFilter {
 public:
  explicit HLTJetMassFilter(const edm::ParameterSet&);
  ~HLTJetMassFilter();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  virtual bool hltFilter( edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs& filterobject) const override;

 private:

  edm::InputTag src_;
  const edm::EDGetTokenT<reco::PFJetCollection> inputPFToken_;
  double minJetMass_;

};

#endif

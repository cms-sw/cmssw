#ifndef HLTcore_HLTEventAnalyzerRAW_h
#define HLTcore_HLTEventAnalyzerRAW_h

/** \class HLTEventAnalyzerRAW
 *
 *  
 *  This class is an EDAnalyzer analyzing the combined HLT information for RAW
 *
 *  $Date: 2012/01/30 09:40:35 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

//
// class declaration
//
class HLTEventAnalyzerRAW : public edm::EDAnalyzer {
  
 public:
  explicit HLTEventAnalyzerRAW(const edm::ParameterSet&);
  ~HLTEventAnalyzerRAW();

  virtual void beginRun(edm::Run const &, edm::EventSetup const&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void analyzeTrigger(const edm::Event&, const edm::EventSetup&, const std::string& triggerName);

 private:

  /// module config parameters
  std::string   processName_;
  std::string   triggerName_;
  edm::InputTag triggerResultsTag_;
  edm::InputTag triggerEventWithRefsTag_;

  /// additional class data memebers
  edm::Handle<edm::TriggerResults>           triggerResultsHandle_;
  edm::Handle<trigger::TriggerEventWithRefs> triggerEventWithRefsHandle_;
  HLTConfigProvider hltConfig_;

  /// payload extracted from TriggerEventWithRefs

  trigger::Vids        photonIds_;
  trigger::VRphoton    photonRefs_;
  trigger::Vids        electronIds_;
  trigger::VRelectron  electronRefs_;
  trigger::Vids        muonIds_;
  trigger::VRmuon      muonRefs_;
  trigger::Vids        jetIds_;
  trigger::VRjet       jetRefs_;
  trigger::Vids        compositeIds_;
  trigger::VRcomposite compositeRefs_;
  trigger::Vids        basemetIds_;
  trigger::VRbasemet   basemetRefs_;
  trigger::Vids        calometIds_;
  trigger::VRcalomet   calometRefs_;
  trigger::Vids        pixtrackIds_;
  trigger::VRpixtrack  pixtrackRefs_;

  trigger::Vids        l1emIds_;
  trigger::VRl1em      l1emRefs_;
  trigger::Vids        l1muonIds_;
  trigger::VRl1muon    l1muonRefs_;
  trigger::Vids        l1jetIds_;
  trigger::VRl1jet     l1jetRefs_;
  trigger::Vids        l1etmissIds_;
  trigger::VRl1etmiss  l1etmissRefs_;
  trigger::Vids        l1hfringsIds_;
  trigger::VRl1hfrings l1hfringsRefs_;

  trigger::Vids        pfjetIds_;
  trigger::VRpfjet     pfjetRefs_;
  trigger::Vids        pftauIds_;
  trigger::VRpftau     pftauRefs_;

};
#endif

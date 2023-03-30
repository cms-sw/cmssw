#ifndef HLTrigger_HLTcore_HLTEventAnalyzerRAW_h
#define HLTrigger_HLTcore_HLTEventAnalyzerRAW_h

/** \class HLTEventAnalyzerRAW
 *
 *
 *  This class is an EDAnalyzer analyzing the combined HLT information for RAW
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//
class HLTEventAnalyzerRAW : public edm::stream::EDAnalyzer<> {
public:
  explicit HLTEventAnalyzerRAW(const edm::ParameterSet&);
  ~HLTEventAnalyzerRAW() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  virtual void analyzeTrigger(const edm::Event&, const edm::EventSetup&, const std::string& triggerName);

private:
  using LOG = edm::LogVerbatim;

  static constexpr const char* logMsgType_ = "HLTEventAnalyzerRAW";

  template <class TVID, class TVREF>
  void showObjects(TVID const& vids, TVREF const& vrefs, std::string const& name) const;

  template <class TREF>
  void showObject(LOG& log, TREF const& ref) const;

  /// module config parameters
  const std::string processName_;
  const std::string triggerName_;
  const edm::InputTag triggerResultsTag_;
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  const edm::InputTag triggerEventWithRefsTag_;
  const edm::EDGetTokenT<trigger::TriggerEventWithRefs> triggerEventWithRefsToken_;

  /// additional class data members
  bool const verbose_;
  bool const permissive_;

  edm::Handle<edm::TriggerResults> triggerResultsHandle_;
  edm::Handle<trigger::TriggerEventWithRefs> triggerEventWithRefsHandle_;

  HLTConfigProvider hltConfig_;

  /// payload extracted from TriggerEventWithRefs
  trigger::Vids photonIds_;
  trigger::VRphoton photonRefs_;
  trigger::Vids electronIds_;
  trigger::VRelectron electronRefs_;
  trigger::Vids muonIds_;
  trigger::VRmuon muonRefs_;
  trigger::Vids jetIds_;
  trigger::VRjet jetRefs_;
  trigger::Vids compositeIds_;
  trigger::VRcomposite compositeRefs_;
  trigger::Vids basemetIds_;
  trigger::VRbasemet basemetRefs_;
  trigger::Vids calometIds_;
  trigger::VRcalomet calometRefs_;
  trigger::Vids pixtrackIds_;
  trigger::VRpixtrack pixtrackRefs_;

  trigger::Vids l1emIds_;
  trigger::VRl1em l1emRefs_;
  trigger::Vids l1muonIds_;
  trigger::VRl1muon l1muonRefs_;
  trigger::Vids l1jetIds_;
  trigger::VRl1jet l1jetRefs_;
  trigger::Vids l1etmissIds_;
  trigger::VRl1etmiss l1etmissRefs_;
  trigger::Vids l1hfringsIds_;
  trigger::VRl1hfrings l1hfringsRefs_;

  trigger::Vids l1tmuonIds_;
  trigger::VRl1tmuon l1tmuonRefs_;
  trigger::Vids l1tmuonShowerIds_;
  trigger::VRl1tmuonShower l1tmuonShowerRefs_;
  trigger::Vids l1tegammaIds_;
  trigger::VRl1tegamma l1tegammaRefs_;
  trigger::Vids l1tjetIds_;
  trigger::VRl1tjet l1tjetRefs_;
  trigger::Vids l1ttauIds_;
  trigger::VRl1ttau l1ttauRefs_;
  trigger::Vids l1tetsumIds_;
  trigger::VRl1tetsum l1tetsumRefs_;

  /// Phase 2
  trigger::Vids l1ttkmuIds_;
  trigger::VRl1ttkmuon l1ttkmuRefs_;
  trigger::Vids l1ttkeleIds_;
  trigger::VRl1ttkele l1ttkeleRefs_;
  trigger::Vids l1ttkemIds_;
  trigger::VRl1ttkem l1ttkemRefs_;
  trigger::Vids l1tpfjetIds_;
  trigger::VRl1tpfjet l1tpfjetRefs_;
  trigger::Vids l1tpftauIds_;
  trigger::VRl1tpftau l1tpftauRefs_;
  trigger::Vids l1thpspftauIds_;
  trigger::VRl1thpspftau l1thpspftauRefs_;
  trigger::Vids l1tpftrackIds_;
  trigger::VRl1tpftrack l1tpftrackRefs_;

  trigger::Vids pfjetIds_;
  trigger::VRpfjet pfjetRefs_;
  trigger::Vids pftauIds_;
  trigger::VRpftau pftauRefs_;
  trigger::Vids pfmetIds_;
  trigger::VRpfmet pfmetRefs_;
};

template <class TVID, class TVREF>
void HLTEventAnalyzerRAW::showObjects(TVID const& vids, TVREF const& vrefs, std::string const& name) const {
  size_t const size = vids.size();
  assert(size == vrefs.size());

  if (size == 0) {
    return;
  }

  LOG(logMsgType_) << "   " << name << ": size=" << size;
  for (size_t idx = 0; idx < size; ++idx) {
    LOG log(logMsgType_);
    log << "    [" << idx << "] id=" << vids[idx] << " ";
    auto const& ref = vrefs[idx];
    if (permissive_ and not ref.isAvailable()) {
      log << "(Ref with id=" << ref.id() << " not available)";
    } else {
      showObject(log, ref);
    }
  }
}

template <class TREF>
void HLTEventAnalyzerRAW::showObject(LOG& log, TREF const& ref) const {
  log << "pt=" << ref->pt() << " eta=" << ref->eta() << " phi=" << ref->phi() << " mass=" << ref->mass();
}

template <>
void HLTEventAnalyzerRAW::showObject(LOG& log, trigger::VRl1hfrings::value_type const& ref) const;

#endif  // HLTrigger_HLTcore_HLTEventAnalyzerRAW_h

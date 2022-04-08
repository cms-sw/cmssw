#ifndef DQM_GEM_GEMEffByGEMCSCSegmentSource_h
#define DQM_GEM_GEMEffByGEMCSCSegmentSource_h

/** \class GEMEffByGEMCSCSegmentSource
 * 
 * `GEMEffByGEMCSCSegmentSource` measures the efficiency of GE11-L1(2) using GE11-L2(1) and ME11 as trigger detectors.
 * See https://github.com/cms-sw/cmssw/blob/CMSSW_12_3_0_pre5/RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegAlgoRR.cc
 *
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/GEMRecHit/interface/GEMCSCSegmentCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class GEMEffByGEMCSCSegmentSource : public DQMEDAnalyzer {
public:
  explicit GEMEffByGEMCSCSegmentSource(const edm::ParameterSet &);
  ~GEMEffByGEMCSCSegmentSource() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  // NOTE
  using MEMap = std::map<GEMDetId, dqm::impl::MonitorElement*>;
  bool hasMEKey(const MEMap&, const GEMDetId&);
  void fillME(dqm::impl::MonitorElement*, const double);
  void fillME(MEMap& me_map, const GEMDetId& key, const double);
  void fillMEWithinLimits(dqm::impl::MonitorElement*, const double);
  void fillMEWithinLimits(MEMap&, const GEMDetId&, const double);
  template <typename T>
  inline bool checkRefs(const std::vector<T*>&);
  inline GEMDetId getReStLaKey(const GEMDetId&);
  const double kEps_ = std::numeric_limits<double>::epsilon();

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

  void bookEfficiencyChamber(DQMStore::IBooker &, const edm::ESHandle<GEMGeometry> &);
  void bookMisc(DQMStore::IBooker &, const edm::ESHandle<GEMGeometry> &);
  MonitorElement *bookNumerator1D(DQMStore::IBooker &, MonitorElement *);

  // ME11-GE11 segments
  void analyzeME11GE11Segment(const GEMCSCSegment &);
  void checkCoincidenceGE11(const GEMRecHit *, const GEMRecHit *, const GEMCSCSegment &);
  void findMatchedME11Segments(const reco::MuonCollection *);
  bool isME11SegmentMatched(const CSCSegment &);

  // const member data (mainly parameters)
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> kGEMTokenBeginRun_;
  const edm::EDGetTokenT<GEMCSCSegmentCollection> kGEMCSCSegmentToken_;
  const edm::EDGetTokenT<reco::MuonCollection> kMuonToken_;
  const bool kUseMuon_;
  const uint32_t kMinCSCRecHits_;
  const std::string kFolder_;
  const std::string kLogCategory_;

  // member data
  std::vector<const CSCSegment *> matched_me11_segment_vector_;

  // MonitorElement
  MEMap me_chamber_;  // 1D, (region, station, layer)
  MEMap me_chamber_matched_;
  MEMap me_muon_chamber_;  // 1D, (region, station, layer)
  MEMap me_muon_chamber_matched_;
  // misc
  MEMap me_num_csc_hits_;
  MEMap me_num_csc_hits_matched_;
  MEMap me_reduced_chi2_;
  MEMap me_reduced_chi2_matched_;
  MEMap me_csc_chamber_type_;
  MEMap me_csc_chamber_type_matched_;
};

template <typename T>
inline bool GEMEffByGEMCSCSegmentSource::checkRefs(const std::vector<T*>& refs) {
  if (refs.empty())
    return false;
  if (refs.front() == nullptr)
    return false;
  return true;
}

inline GEMDetId GEMEffByGEMCSCSegmentSource::getReStLaKey(const GEMDetId& id) {
  return GEMDetId{id.region(), 1, id.station(), id.layer(), 0, 0};
}

#endif  // DQM_GEM_GEMEffByGEMCSCSegmentSource_h

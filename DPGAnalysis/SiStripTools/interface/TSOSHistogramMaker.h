#ifndef TRACKRECOMONITOR_TSOSHISTOGRAMMAKER_H
#define TRACKRECOMONITOR_TSOSHISTOGRAMMAKER_H

#include <vector>
#include <string>
#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class TH1F;
class TH2F;
class TrajectoryStateOnSurface;
namespace edm {
  class ParameterSet;
}
//class TransientTrackingRecHit { public: class ConstRecHitPointer;};
class TSOSHistogramMaker {
public:
  TSOSHistogramMaker();
  TSOSHistogramMaker(const edm::ParameterSet& iConfig);
  void fill(const TrajectoryStateOnSurface& tsos, TransientTrackingRecHit::ConstRecHitPointer hit) const;

private:
  const bool m_2dhistos;
  std::vector<DetIdSelector> m_detsels;
  std::vector<std::string> m_selnames;
  std::vector<std::string> m_seltitles;

  std::vector<TH2F*> m_histocluslenangle;
  std::vector<TH1F*> m_tsosy;
  std::vector<TH1F*> m_tsosx;
  std::vector<TH2F*> m_tsosxy;
  std::vector<TH1F*> m_tsosprojx;
  std::vector<TH1F*> m_tsosprojy;
  std::vector<TH1F*> m_ttrhy;
  std::vector<TH1F*> m_ttrhx;
  std::vector<TH2F*> m_ttrhxy;
  std::vector<TH1F*> m_tsosdy;
  std::vector<TH1F*> m_tsosdx;
  std::vector<TH2F*> m_tsosdxdy;
};

#endif  // TRACKRECOMONITOR_TSOSHISTOGRAMMAKER_H

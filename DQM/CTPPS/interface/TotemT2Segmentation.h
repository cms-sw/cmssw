/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Author:
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef DQM_CTPPS_TotemT2Segmentation_h
#define DQM_CTPPS_TotemT2Segmentation_h

#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"
#include "Geometry/ForwardGeometry/interface/TotemGeometry.h"

#include <unordered_map>
#include <vector>

class TotemGeometry;
class TH2D;

class TotemT2Segmentation {
public:
  explicit TotemT2Segmentation(const TotemGeometry&, size_t, size_t);

  void fill(TH2D*, const TotemT2DetId&, double value = 1.);

private:
  std::vector<std::pair<short, short> > computeBins(const TotemT2DetId& detid) const;

  const TotemGeometry geom_;
  const size_t nbinsx_;
  const size_t nbinsy_;

  std::unordered_map<TotemT2DetId, std::vector<std::pair<short, short> > > bins_map_;
};

#endif

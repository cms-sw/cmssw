
/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Author:
 *   Laurent Forthomme
 *   Arkadiusz Cwikla
 *
 ****************************************************************************/

#include "DQM/CTPPS/interface/TotemT2Segmentation.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TH2D.h"

TotemT2Segmentation::TotemT2Segmentation(size_t nbinsx, size_t nbinsy) : nbinsx_(nbinsx), nbinsy_(nbinsy) {
  for (unsigned short arm = 0; arm <= CTPPSDetId::maxArm; ++arm)
    for (unsigned short plane = 0; plane <= TotemT2DetId::maxPlane; ++plane)
      for (unsigned short id = 0; id <= TotemT2DetId::maxChannel; ++id) {
        const TotemT2DetId detid(arm, plane, id);
        bins_map_[detid] = computeBins(detid);
      }
}

void TotemT2Segmentation::fill(TH2D* hist, const TotemT2DetId& detid, double value) {
  if (bins_map_.count(detid) == 0)
    throw cms::Exception("TotemT2Segmentation") << "Failed to retrieve list of bins for TotemT2DetId " << detid << ".";
  if (static_cast<size_t>(hist->GetXaxis()->GetNbins()) != nbinsx_ ||
      static_cast<size_t>(hist->GetYaxis()->GetNbins()) != nbinsy_)
    throw cms::Exception("TotemT2Segmentation")
        << "Trying to fill a summary plot with invalid number of bins. "
        << "Should be of dimension (" << nbinsx_ << ", " << nbinsy_ << "), but has dimension ("
        << hist->GetXaxis()->GetNbins() << ", " << hist->GetYaxis()->GetNbins() << ").";
  for (const auto& bin : bins_map_.at(detid))
    hist->Fill(bin.first, bin.second, value);
}

std::vector<std::pair<short, short> > TotemT2Segmentation::computeBins(const TotemT2DetId& detid) const {
  std::vector<std::pair<short, short> > bins;
  // find the histogram centre
  const auto ox = floor(nbinsx_ * 0.5), oy = floor(nbinsy_ * 0.5);
  // compute the ellipse parameters
  const auto ax = ceil(nbinsx_ * 0.5), by = ceil(nbinsy_ * 0.5);

  const float max_half_angle_rad = 0.3;

  // Without use of the Totem Geometry, follow an octant-division of channels,
  // with even and odd planes alternating around the circle. Starting at 9AM
  // and going clockwise around in increments of 1h30min=45deg, we have
  // Ch0 on even planes, Ch0+odd, Ch1+even, Ch1+odd, up to Ch3+odd at 7:30PM

  const float tile_angle_rad = (180 - 45. / 2 - (detid.plane() % 2 ? 45 : 0) - (detid.channel() * 90)) * M_PI / 180.;
  // Geometric way of associating a DetId to a vector<ix, iy> of bins given the size (nx_, ny_) of
  // the TH2D('s) to be filled
  for (size_t ix = 0; ix < nbinsx_; ++ix)
    for (size_t iy = 0; iy < nbinsy_; ++iy) {
      const auto ell_rad_norm = std::pow((ix - ox) / ax, 2) + std::pow((iy - oy) / by, 2);
      if (ell_rad_norm < 1. && ell_rad_norm >= 0.1 &&
          fabs(std::atan2(iy - oy, ix - ox) - tile_angle_rad) < max_half_angle_rad)
        bins.emplace_back(ix, iy);
    }

  return bins;
}

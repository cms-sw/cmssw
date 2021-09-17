#include <cmath>
#include <cassert>

#include "RecoJets/FFTJetAlgorithms/interface/LookupTable2d.h"

namespace fftjetcms {
  LookupTable2d::LookupTable2d(
      unsigned nx, double xmin, double xmax, unsigned ny, double ymin, double ymax, const std::vector<double>& data)
      : data_(data),
        nx_(nx),
        ny_(ny),
        xmin_(xmin),
        xmax_(xmax),
        ymin_(ymin),
        ymax_(ymax),
        bwx_((xmax - xmin) / nx),
        bwy_((ymax - ymin) / ny) {
    assert(nx_);
    assert(ny_);
    assert(xmin_ < xmax_);
    assert(ymin_ < ymax_);
    assert(data_.size() == nx_ * ny_);
  }

  double LookupTable2d::closest(const double x, const double y) const {
    const unsigned ix = x <= xmin_                ? 0U
                        : x >= xmax_ - bwx_ / 2.0 ? nx_ - 1U
                                                  : static_cast<unsigned>((x - xmin_) / bwx_);
    const unsigned iy = y <= ymin_                ? 0U
                        : y >= ymax_ - bwy_ / 2.0 ? ny_ - 1U
                                                  : static_cast<unsigned>((y - ymin_) / bwy_);
    return data_[ix * ny_ + iy];
  }
}  // namespace fftjetcms

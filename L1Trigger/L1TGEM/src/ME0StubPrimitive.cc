#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"

//define class ME0StubPrimitive
ME0StubPrimitive::ME0StubPrimitive() : layerCount_{0}, hitCount_{0}, patternId_{0}, strip_{0}, etaPartition_{0} {
  updateQuality();
}
ME0StubPrimitive::ME0StubPrimitive(int layerCount, int hitCount, int patternId, int strip, int etaPartition)
    : layerCount_{layerCount}, hitCount_{hitCount}, patternId_{patternId}, strip_{strip}, etaPartition_{etaPartition} {
  updateQuality();
}
ME0StubPrimitive::ME0StubPrimitive(int layerCount, int hitCount, int patternId, int strip, int etaPartition, double bx)
    : layerCount_{layerCount},
      hitCount_{hitCount},
      patternId_{patternId},
      strip_{strip},
      etaPartition_{etaPartition},
      bx_{bx} {
  updateQuality();
}
ME0StubPrimitive::ME0StubPrimitive(
    int layerCount, int hitCount, int patternId, int strip, int etaPartition, double bx, std::vector<int>& centroids)
    : layerCount_{layerCount},
      hitCount_{hitCount},
      patternId_{patternId},
      strip_{strip},
      etaPartition_{etaPartition},
      bx_{bx},
      centroids_{centroids} {
  updateQuality();
}
void ME0StubPrimitive::reset() {
  layerCount_ = 0;
  hitCount_ = 0;
  patternId_ = 0;
  updateQuality();
}
void ME0StubPrimitive::updateQuality() {
  int idMask;
  if (layerCount_) {
    if (ignoreBend_) {
      idMask = 0xfe;
    } else {
      idMask = 0xff;
    }
    quality_ = (layerCount_ << 23) | (hitCount_ << 17) | ((patternId_ & idMask) << 12) | (strip_ << 4) | etaPartition_;
  } else {
    quality_ = 0;
  }
}
void ME0StubPrimitive::fit(int patSpan) {
  if (patternId_ != 0) {
    // std::vector<double> tmp;
    // tmp.reserve(centroids_.size());
    // for (double centroid : centroids_) {
    //   tmp.push_back(centroid - (patSpan / 2 + 1));
    // }
    // std::vector<double> x;
    // std::vector<double> centroids;
    // for (uint32_t i = 0; i < tmp.size(); ++i) {
    //   if (tmp[i] != -1 * (maxSpan / 2 + 1)) {
    //     x.push_back(i - 2.5);
    //     centroids.push_back(tmp[i]);
    //   }
    // }
    // std::vector<double> fit = llseFit(x, centroids);
    // bendingAngle_ = fit[0];
    // subStrip_ = fit[1];
    // mse_ = fit[2];
    std::vector<bool> validMask;
    validMask.reserve(centroids_.size());
    for (int centroid : centroids_) {
      validMask.push_back(centroid > 0);
    }
    std::vector<double> fit_ = l1t::me0::vhdlExactFit(centroids_, validMask);
    bendingAngle_ = fit_[0] / 2.0;                  // m
    subStrip_ = (fit_[2] / 2.0) - patSpan / 2 - 1;  // b
  }
}

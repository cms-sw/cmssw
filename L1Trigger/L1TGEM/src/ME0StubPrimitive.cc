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
    int layerCount, int hitCount, int patternId, int strip, int etaPartition, double bx, std::vector<double>& centroids)
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
void ME0StubPrimitive::fit(int maxSpan) {
  if (patternId_ != 0) {
    std::vector<double> tmp;
    tmp.reserve(centroids_.size());
    for (double centroid : centroids_) {
      tmp.push_back(centroid - (maxSpan / 2 + 1));
    }
    std::vector<double> x;
    std::vector<double> centroids;
    for (uint32_t i = 0; i < tmp.size(); ++i) {
      if (tmp[i] != -1 * (maxSpan / 2 + 1)) {
        x.push_back(i - 2.5);
        centroids.push_back(tmp[i]);
      }
    }
    std::vector<double> fit = llseFit(x, centroids);
    bendingAngle_ = fit[0];
    subStrip_ = fit[1];
    mse_ = fit[2];
  }
}
std::vector<double> ME0StubPrimitive::llseFit(const std::vector<double>& x, const std::vector<double>& y) {
  double xSum = 0;
  double ySum = 0;
  for (double val : x) {
    xSum += val;
  }
  for (double val : y) {
    ySum += val;
  }
  int n = x.size();
  // linear regression
  double product = 0;
  double squares = 0;
  for (int i = 0; i < n; ++i) {
    product += (n * x[i] - xSum) * (n * y[i] - ySum);
    squares += (n * x[i] - xSum) * (n * x[i] - xSum);
  }

  double m = product / squares;
  double b = (ySum - m * xSum) / n;
  double sse = 0.0;
  for (int i = 0; i < n; ++i) {
    sse += (y[i] - m * x[i] - b) * (y[i] - m * x[i] - b);
  }

  std::vector<double> fit = {m, b, sse / n};
  return fit;
}

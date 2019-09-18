#include "CondFormats/SiStripObjects/interface/SiStripApvSimulationParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Random/RandFlat.h"

namespace {
  PhysicsTools::Calibration::HistogramF2D calculateXInt(const SiStripApvSimulationParameters::LayerParameters& params) {
    auto hXInt = (params.hasEquidistantBinsY()
                      ? (params.hasEquidistantBinsZ()
                             ? PhysicsTools::Calibration::HistogramF2D(
                                   params.numberOfBinsY(), params.rangeY(), params.numberOfBinsZ(), params.rangeZ())
                             : PhysicsTools::Calibration::HistogramF2D(
                                   params.numberOfBinsY(), params.rangeY(), params.upperLimitsZ()))
                      : (params.hasEquidistantBinsZ()
                             ? PhysicsTools::Calibration::HistogramF2D(
                                   params.upperLimitsY(), params.numberOfBinsZ(), params.rangeZ())
                             : PhysicsTools::Calibration::HistogramF2D(params.upperLimitsY(), params.upperLimitsZ())));
    for (int i{0}; i != params.numberOfBinsY() + 2; ++i) {
      for (int j{0}; j != params.numberOfBinsZ() + 2; ++j) {
        float xInt = 0.;
        for (int k{0}; k != params.numberOfBinsX() + 2; ++k) {
          xInt += params.binContent(k, i, j);
        }
        hXInt.setBinContent(i, j, xInt);
      }
    }
    return hXInt;
  }

  float xBinPos(const SiStripApvSimulationParameters::LayerParameters& hist, int iBin, float pos = 0.5) {
    // NOTE: does not work for under- and overflow bins (iBin = 0 and iBIn == hist.numberOfBinsX()+1)
    if (hist.hasEquidistantBinsX()) {
      const auto range = hist.rangeX();
      const auto binWidth = (range.max - range.min) / hist.numberOfBinsX();
      return range.min + (iBin - 1 + pos) * binWidth;
    } else {
      return (1. - pos) * hist.upperLimitsX()[iBin - 1] + pos * hist.upperLimitsX()[iBin];
    }
  }
}  // namespace

void SiStripApvSimulationParameters::calculateIntegrals() {
  if (m_barrelParam.size() != m_barrelParam_xInt.size()) {
    m_barrelParam_xInt.resize(m_barrelParam.size());
    for (unsigned int i{0}; i != m_barrelParam.size(); ++i) {
      m_barrelParam_xInt[i] = calculateXInt(m_barrelParam[i]);
    }
  }
  if (m_endcapParam.size() != m_endcapParam_xInt.size()) {
    m_endcapParam_xInt.resize(m_endcapParam.size());
    for (unsigned int i{0}; i != m_endcapParam.size(); ++i) {
      m_endcapParam_xInt[i] = calculateXInt(m_endcapParam[i]);
    }
  }
}

bool SiStripApvSimulationParameters::putTIB(SiStripApvSimulationParameters::layerid layer,
                                            SiStripApvSimulationParameters::LayerParameters&& params) {
  if ((layer > m_nTIB) || (layer < 1)) {
    edm::LogError("SiStripApvSimulationParameters")
        << "[" << __PRETTY_FUNCTION__ << "] layer index " << layer << " out of range [1," << m_nTIB << "]";
    return false;
  }
  m_barrelParam[layer - 1] = params;
  m_barrelParam_xInt[layer - 1] = calculateXInt(params);
  return true;
}

bool SiStripApvSimulationParameters::putTOB(SiStripApvSimulationParameters::layerid layer,
                                            SiStripApvSimulationParameters::LayerParameters&& params) {
  if ((layer > m_nTOB) || (layer < 1)) {
    edm::LogError("SiStripApvSimulationParameters")
        << "[" << __PRETTY_FUNCTION__ << "] layer index " << layer << " out of range [1," << m_nTOB << ")";
    return false;
  }
  m_barrelParam[m_nTIB + layer - 1] = params;
  m_barrelParam_xInt[m_nTIB + layer - 1] = calculateXInt(params);
  return true;
}

bool SiStripApvSimulationParameters::putTID(SiStripApvSimulationParameters::layerid wheel,
                                            SiStripApvSimulationParameters::LayerParameters&& params) {
  if ((wheel > m_nTID) || (wheel < 1)) {
    edm::LogError("SiStripApvSimulationParameters")
        << "[" << __PRETTY_FUNCTION__ << "] wheel index " << wheel << " out of range [1," << m_nTID << "]";
    return false;
  }
  m_endcapParam[wheel - 1] = params;
  m_endcapParam_xInt[wheel - 1] = calculateXInt(params);
  return true;
}

bool SiStripApvSimulationParameters::putTEC(SiStripApvSimulationParameters::layerid wheel,
                                            SiStripApvSimulationParameters::LayerParameters&& params) {
  if ((wheel > m_nTEC) || (wheel < 1)) {
    edm::LogError("SiStripApvSimulationParameters")
        << "[" << __PRETTY_FUNCTION__ << "] wheel index " << wheel << " out of range [1," << m_nTEC << ")";
    return false;
  }
  m_endcapParam[m_nTID + wheel - 1] = params;
  m_endcapParam_xInt[m_nTID + wheel - 1] = calculateXInt(params);
  return true;
}

float SiStripApvSimulationParameters::sampleBarrel(layerid layerIdx,
                                                   float z,
                                                   float pu,
                                                   CLHEP::HepRandomEngine* engine) const {
  if (m_barrelParam.size() != m_barrelParam_xInt.size()) {
    throw cms::Exception("LogicError") << "x-integrals of 3D histograms have not been calculated";
  }
  const auto layerParam = m_barrelParam[layerIdx];
  const int ip = layerParam.findBinY(pu);
  const int iz = layerParam.findBinZ(z);
  const float norm = m_barrelParam_xInt[layerIdx].binContent(ip, iz);
  const auto val = CLHEP::RandFlat::shoot(engine) * norm;
  if (val < layerParam.binContent(0, ip, iz)) {  // underflow
    return layerParam.rangeX().min;
  } else if (norm - val < layerParam.binContent(layerParam.numberOfBinsX() + 1, ip, iz)) {  // overflow
    return layerParam.rangeX().max;
  } else {  // loop over bins, return center of our bin
    float sum = layerParam.binContent(0, ip, iz);
    for (int i{1}; i != layerParam.numberOfBinsX() + 1; ++i) {
      sum += layerParam.binContent(i, ip, iz);
      if (sum > val) {
        return xBinPos(layerParam, i, (sum - val) / layerParam.binContent(i, ip, iz));
      }
    }
  }
  throw cms::Exception("LogicError") << "Problem drawing a random number from the distribution";
}

float SiStripApvSimulationParameters::sampleEndcap(layerid wheelIdx,
                                                   float r,
                                                   float pu,
                                                   CLHEP::HepRandomEngine* engine) const {
  if (m_endcapParam.size() != m_endcapParam_xInt.size()) {
    throw cms::Exception("LogicError") << "x-integrals of 3D histograms have not been calculated";
  }
  const auto layerParam = m_endcapParam[wheelIdx];
  const int ip = layerParam.findBinY(pu);
  const int ir = layerParam.findBinZ(r);
  const float norm = m_endcapParam_xInt[wheelIdx].binContent(ip, ir);
  const auto val = CLHEP::RandFlat::shoot(engine) * norm;
  if (val < layerParam.binContent(0, ip, ir)) {  // underflow
    return layerParam.rangeX().min;
  } else if (norm - val < layerParam.binContent(layerParam.numberOfBinsX() + 1, ip, ir)) {  // overflow
    return layerParam.rangeX().max;
  } else {  // loop over bins, return center of our bin
    float sum = layerParam.binContent(0, ip, ir);
    for (int i{1}; i != layerParam.numberOfBinsX() + 1; ++i) {
      sum += layerParam.binContent(i, ip, ir);
      if (sum > val) {
        return xBinPos(layerParam, i, (sum - val) / layerParam.binContent(i, ip, ir));
      }
    }
  }
  throw cms::Exception("LogicError") << "Problem drawing a random number from the distribution";
}

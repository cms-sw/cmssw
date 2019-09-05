#ifndef SiStripApvSimulationParameters_h
#define SiStripApvSimulationParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"

namespace CLHEP {
  class HepRandomEngine;
}

/**
 * Stores a histogram binned in PU, z and baseline voltage, for every barrel layer of the strip tracker
 */
class SiStripApvSimulationParameters {
public:
  using layerid = unsigned int;
  using LayerParameters = PhysicsTools::Calibration::HistogramF3D;

  SiStripApvSimulationParameters(layerid nTIB, layerid nTOB) : m_nTIB(nTIB), m_nTOB(nTOB) {
    m_barrelParam.resize(m_nTIB + m_nTOB);
    m_barrelParam_xInt.resize(m_nTIB + m_nTOB);
  }
  SiStripApvSimulationParameters() {}
  ~SiStripApvSimulationParameters() {}

  void calculateIntegrals();  // make sure integrals have been calculated

  bool putTIB(layerid layer, const LayerParameters& params) { return putTIB(layer, LayerParameters(params)); }
  bool putTIB(layerid layer, LayerParameters&& params);

  bool putTOB(layerid layer, const LayerParameters& params) { return putTOB(layer, LayerParameters(params)); }
  bool putTOB(layerid layer, LayerParameters&& params);

  const LayerParameters& getTIB(layerid layer) const { return m_barrelParam[layer - 1]; }
  const LayerParameters& getTOB(layerid layer) const { return m_barrelParam[m_nTIB + layer - 1]; }

  float sampleTIB(layerid layer, float z, float pu, CLHEP::HepRandomEngine* engine) const {
    return sampleBarrel(layer - 1, z, pu, engine);
  }
  float sampleTOB(layerid layer, float z, float pu, CLHEP::HepRandomEngine* engine) const {
    return sampleBarrel(m_nTIB + layer - 1, z, pu, engine);
  };

private:
  layerid m_nTIB, m_nTOB;
  std::vector<PhysicsTools::Calibration::HistogramF3D> m_barrelParam;
  std::vector<PhysicsTools::Calibration::HistogramF2D> m_barrelParam_xInt;

  float sampleBarrel(layerid layerIdx, float z, float pu, CLHEP::HepRandomEngine* engine) const;

  COND_SERIALIZABLE;
};

#endif

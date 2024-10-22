#ifndef Geometry_ForwardGeometry_ZdcHardcodeGeometryEP_H
#define Geometry_ForwardGeometry_ZdcHardcodeGeometryEP_H 1

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/ZDCGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "Geometry/ForwardGeometry/interface/ZdcHardcodeGeometryLoader.h"

//
// class decleration
//

class ZdcHardcodeGeometryEP : public edm::ESProducer {
public:
  ZdcHardcodeGeometryEP(const edm::ParameterSet&);
  ~ZdcHardcodeGeometryEP() override;

  using ReturnType = std::unique_ptr<CaloSubdetectorGeometry>;

  ReturnType produce(const ZDCGeometryRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  // ----------member data ---------------------------

  edm::ESGetToken<ZdcTopology, HcalRecNumberingRecord> m_zdcTopoToken;
  std::unique_ptr<ZdcHardcodeGeometryLoader> m_loader;

  bool m_applyAlignment;
  bool m_zdcAddRPD;
};

#endif

// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

#ifndef __RecoHGCal_TICL_SeedingRegionGlobal_H__
#define __RecoHGCal_TICL_SeedingRegionGlobal_H__
#include <memory>  // unique_ptr
#include <string>
#include "RecoHGCal/TICL/interface/SeedingRegionAlgoBase.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"



class HGCGraph;

namespace ticl {
  class SeedingRegionGlobal final : public SeedingRegionAlgoBase {
  public:
    SeedingRegionGlobal(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes);
    ~SeedingRegionGlobal() override;

    void makeRegions(const edm::Event& ev,
		     const edm::EventSetup& es,
		     std::vector<ticl::TICLSeedingRegion>& result) override;

  private:

  };
}  // namespace ticl
#endif

// Author: dalfonso@cern.ch
// Date: 02/2021

#ifndef RecoHGCal_TICL_SeedingRegionByHF_h
#define RecoHGCal_TICL_SeedingRegionByHF_h
#include <memory>  // unique_ptr
#include <string>
#include "RecoHGCal/TICL/plugins/SeedingRegionAlgoBase.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

namespace ticl {
  class SeedingRegionByHF final : public SeedingRegionAlgoBase {
  public:
    SeedingRegionByHF(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes);
    ~SeedingRegionByHF() override;

    void initialize(const edm::EventSetup& es) override;

    void makeRegions(const edm::Event& ev, const edm::EventSetup& es, std::vector<TICLSeedingRegion>& result) override;
    static void fillPSetDescription(edm::ParameterSetDescription& desc);
    static edm::ParameterSetDescription makePSetDescription();

  private:
    void buildFirstLayers();

    edm::EDGetTokenT<HFRecHitCollection> hfhits_token_;

    int algoVerbosity_ = 0;

    double minAbsEta_;
    double maxAbsEta_;
    double minEt_;

    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geo_token_;
    const CaloGeometry* geometry_;
  };
}  // namespace ticl
#endif

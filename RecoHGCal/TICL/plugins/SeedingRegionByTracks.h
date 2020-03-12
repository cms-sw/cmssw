// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

#ifndef RecoHGCal_TICL_SeedingRegionByTracks_h
#define RecoHGCal_TICL_SeedingRegionByTracks_h
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
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class HGCGraph;

namespace ticl {
  class SeedingRegionByTracks final : public SeedingRegionAlgoBase {
  public:
    SeedingRegionByTracks(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes);
    ~SeedingRegionByTracks() override;

    void initialize(const edm::EventSetup& es) override;

    void makeRegions(const edm::Event& ev, const edm::EventSetup& es, std::vector<TICLSeedingRegion>& result) override;

  private:
    void buildFirstLayers();

    edm::EDGetTokenT<reco::TrackCollection> tracks_token_;
    std::once_flag initializeGeometry_;
    const HGCalDDDConstants* hgcons_;
    const StringCutObjectSelector<reco::Track> cutTk_;
    inline static const std::string detectorName_ = "HGCalEESensitive";
    edm::ESHandle<Propagator> propagator_;
    const std::string propName_;
    edm::ESHandle<MagneticField> bfield_;
    std::unique_ptr<GeomDet> firstDisk_[2];
  };
}  // namespace ticl
#endif

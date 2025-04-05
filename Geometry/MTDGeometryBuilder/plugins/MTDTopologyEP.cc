//#define EDM_ML_DEBUG

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"

#include <memory>

class MTDTopologyEP : public edm::ESProducer {
public:
  MTDTopologyEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<MTDTopology>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  ReturnType produce(const MTDTopologyRcd&);

private:
  void fillBTLtopology(const MTDGeometry&, MTDTopology::BTLValues&);
  void fillETLtopology(const PMTDParameters&, int& mtdTopologyMode, MTDTopology::ETLValues&);

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<PMTDParameters, PMTDParametersRcd> mtdparToken_;
};

MTDTopologyEP::MTDTopologyEP(const edm::ParameterSet& conf) {
  auto cc = setWhatProduced(this);
  mtdgeoToken_ = cc.consumesFrom<MTDGeometry, MTDDigiGeometryRecord>(edm::ESInputTag());
  mtdparToken_ = cc.consumesFrom<PMTDParameters, PMTDParametersRcd>(edm::ESInputTag());
}

void MTDTopologyEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription ttc;
  descriptions.add("mtdTopology", ttc);
}

MTDTopologyEP::ReturnType MTDTopologyEP::produce(const MTDTopologyRcd& iRecord) {
  int mtdTopologyMode;
  MTDTopology::BTLValues btlVals;
  MTDTopology::ETLValues etlVals;

  // build BTL topology content from MTDGeometry

  fillBTLtopology(iRecord.get(mtdgeoToken_), btlVals);

  // build ETL topology and topology mode information from PMTDParameters

  fillETLtopology(iRecord.get(mtdparToken_), mtdTopologyMode, etlVals);

  return std::make_unique<MTDTopology>(mtdTopologyMode, btlVals, etlVals);
}

void MTDTopologyEP::fillBTLtopology(const MTDGeometry& mtdgeo, MTDTopology::BTLValues& btlVals) {
  MTDTopology::BTLLayout tmpLayout;
  uint32_t index(0), iphi(1), ieta(0);
  if (mtdgeo.detsBTL().size() != tmpLayout.nBTLmodules_) {
    throw cms::Exception("MTDTopologyEP") << "Inconsistent size of BTL structure arrays";
  }
  for (const auto& det : mtdgeo.detsBTL()) {
    ieta++;

    tmpLayout.btlDetId_[index] = det->geographicalId().rawId();
    tmpLayout.btlPhi_[index] = iphi;
    tmpLayout.btlEta_[index] = ieta;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MTDTopologyEP") << "MTDTopology BTL# " << index << " id= " << det->geographicalId().rawId()
                                      << " iphi/ieta= " << iphi << " / " << ieta;
#endif
    if (ieta == tmpLayout.nBTLeta_) {
      iphi++;
      ieta = 0;
    }
    index++;
  }
  btlVals = tmpLayout;
}

void MTDTopologyEP::fillETLtopology(const PMTDParameters& ptp, int& mtdTopologyMode, MTDTopology::ETLValues& etlVals) {
  mtdTopologyMode = ptp.topologyMode_;

  // Check on the internal consistency of thr ETL layout information provided by parameters

  for (size_t it = 3; it <= 9; it++) {
    bool exception = ((MTDTopologyMode::etlLayoutFromTopoMode(mtdTopologyMode) == MTDTopologyMode::EtlLayout::v10) &&
                      (it == 5 || it == 9));
    if (ptp.vitems_[it].vpars_.size() != ptp.vitems_[2].vpars_.size()) {
      if (!exception) {
        throw cms::Exception("MTDTopologyEP") << "Inconsistent size of ETL structure arrays";
      } else {
        LogDebug("MTDTopologyEP") << "Building ETL topology for scenario 1.7";
      }
    }
  }

  MTDTopology::ETLfaceLayout tmpFace;

  if (static_cast<int>(MTDTopologyMode::etlLayoutFromTopoMode(mtdTopologyMode)) <=
      static_cast<int>(MTDTopologyMode::EtlLayout::v8)) {
    // Disc1 Front Face (0), starting with type Right (2)

    tmpFace.idDiscSide_ = 0;  // ETL front side, Disc1
    tmpFace.idDetType1_ = 2;  // ETL module type right

    tmpFace.start_copy_[0] = ptp.vitems_[3].vpars_;  // start_copy_FR
    tmpFace.start_copy_[1] = ptp.vitems_[2].vpars_;  // start_copy_FL
    tmpFace.offset_[0] = ptp.vitems_[7].vpars_;      // offset_FR
    tmpFace.offset_[1] = ptp.vitems_[6].vpars_;      // offset_FL

    etlVals.emplace_back(tmpFace);

    // Disc1 Back Face (1), starting with type Left (1)

    tmpFace.idDiscSide_ = 1;  // ETL back side, Disc1
    tmpFace.idDetType1_ = 1;  // ETL module type left

    tmpFace.start_copy_[0] = ptp.vitems_[4].vpars_;  // start_copy_BL
    tmpFace.start_copy_[1] = ptp.vitems_[5].vpars_;  // start_copy_BR
    tmpFace.offset_[0] = ptp.vitems_[8].vpars_;      // offset_BL
    tmpFace.offset_[1] = ptp.vitems_[9].vpars_;      // offset_BR

    etlVals.emplace_back(tmpFace);

    // Disc2 Front Face (0), starting with type Right (2)

    tmpFace.idDiscSide_ = 2;  // ETL front side, Disc2
    tmpFace.idDetType1_ = 2;  // ETL module type right

    tmpFace.start_copy_[0] = ptp.vitems_[3].vpars_;  // start_copy_FR
    tmpFace.start_copy_[1] = ptp.vitems_[2].vpars_;  // start_copy_FL
    tmpFace.offset_[0] = ptp.vitems_[7].vpars_;      // offset_FR
    tmpFace.offset_[1] = ptp.vitems_[6].vpars_;      // offset_FL

    etlVals.emplace_back(tmpFace);

    // Disc2 Back Face (1), starting with type Left (1)

    tmpFace.idDiscSide_ = 3;  // ETL back side, Disc2
    tmpFace.idDetType1_ = 1;  // ETL module type left

    tmpFace.start_copy_[0] = ptp.vitems_[4].vpars_;  // start_copy_BL
    tmpFace.start_copy_[1] = ptp.vitems_[5].vpars_;  // start_copy_BR
    tmpFace.offset_[0] = ptp.vitems_[8].vpars_;      // offset_BL
    tmpFace.offset_[1] = ptp.vitems_[9].vpars_;      // offset_BR

    etlVals.emplace_back(tmpFace);

  } else if (static_cast<int>(MTDTopologyMode::etlLayoutFromTopoMode(mtdTopologyMode)) >
             static_cast<int>(MTDTopologyMode::EtlLayout::v8)) {
    // Disc1 Front Face (0), starting with type Right (2)

    tmpFace.idDiscSide_ = 0;  // ETL front side, Disc1
    tmpFace.idDetType1_ = 2;  // etl module type HalfFront2

    tmpFace.start_copy_[0] = ptp.vitems_[2].vpars_;  // start_copy_FR
    tmpFace.start_copy_[1] = ptp.vitems_[2].vpars_;  // start_copy_FL
    tmpFace.offset_[0] = ptp.vitems_[6].vpars_;      // offset_FR
    tmpFace.offset_[1] = ptp.vitems_[6].vpars_;      // offset_FL

    etlVals.emplace_back(tmpFace);

    // Disc1 Back Face (1), starting with type Left (1)

    tmpFace.idDiscSide_ = 1;  // ETL back side, Disc1
    tmpFace.idDetType1_ = 2;  // ETL module type HalfBack2

    tmpFace.start_copy_[0] = ptp.vitems_[3].vpars_;  // start_copy_BL
    tmpFace.start_copy_[1] = ptp.vitems_[3].vpars_;  // start_copy_BR
    tmpFace.offset_[0] = ptp.vitems_[7].vpars_;      // offset_BL
    tmpFace.offset_[1] = ptp.vitems_[7].vpars_;      // offset_BR

    etlVals.emplace_back(tmpFace);

    // Disc2 Front Face (0), starting with type Right (2)

    tmpFace.idDiscSide_ = 2;  // ETL front side, Disc2
    tmpFace.idDetType1_ = 2;  // etl module type HalfFront2

    tmpFace.start_copy_[0] = ptp.vitems_[4].vpars_;  // start_copy_FR
    tmpFace.start_copy_[1] = ptp.vitems_[4].vpars_;  // start_copy_FL
    tmpFace.offset_[0] = ptp.vitems_[8].vpars_;      // offset_FR
    tmpFace.offset_[1] = ptp.vitems_[8].vpars_;      // offset_FL

    etlVals.emplace_back(tmpFace);

    // Disc2 Back Face (1), starting with type Left (1)

    tmpFace.idDiscSide_ = 3;  // ETL back side, Disc2
    tmpFace.idDetType1_ = 2;  // ETL module type HalfBack2

    tmpFace.start_copy_[0] = ptp.vitems_[5].vpars_;  // start_copy_BL
    tmpFace.start_copy_[1] = ptp.vitems_[5].vpars_;  // start_copy_BR
    tmpFace.offset_[0] = ptp.vitems_[9].vpars_;      // offset_BL
    tmpFace.offset_[1] = ptp.vitems_[9].vpars_;      // offset_BR

    etlVals.emplace_back(tmpFace);
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MTDTopologyEP") << " Topology mode = " << mtdTopologyMode << "\n";
  auto print_array = [&](std::vector<int> vector) {
    std::stringstream ss;
    for (auto const& elem : vector) {
      ss << " " << elem;
    }
    ss << "\n";
    return ss.str();
  };

  for (auto const& ilay : etlVals) {
    edm::LogVerbatim("MTDTopologyEP") << " disc face = " << ilay.idDiscSide_ << " start det type = " << ilay.idDetType1_
                                      << "\n start_copy[0]= " << print_array(ilay.start_copy_[0])
                                      << "\n start_copy[1]= " << print_array(ilay.start_copy_[1])
                                      << "\n offset[0]= " << print_array(ilay.offset_[0])
                                      << "\n offset[1]= " << print_array(ilay.offset_[1]);
  }

#endif
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDTopologyEP);

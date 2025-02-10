// system include files
#include <cstdio>
#include <iostream>
#include <sys/time.h>

// user include files
#include "CondFormats/DataRecord/interface/SiPixelDynamicInefficiencyRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//
//
// class decleration
//
class SiPixelDynamicInefficiencyReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelDynamicInefficiencyReader(const edm::ParameterSet&);
  ~SiPixelDynamicInefficiencyReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken;
  const edm::ESGetToken<SiPixelDynamicInefficiency, SiPixelDynamicInefficiencyRcd> dynIneffToken;

  const bool printdebug_;
  double thePixelEfficiency[20];                     // Single pixel effciency
  double thePixelColEfficiency[20];                  // Column effciency
  double thePixelChipEfficiency[20];                 // ROC efficiency
  std::vector<double> theLadderEfficiency_BPix[20];  // Ladder efficiency
  std::vector<double> theModuleEfficiency_BPix[20];  // Module efficiency
  std::vector<double> thePUEfficiency[20];           // Instlumi dependent efficiency
  double theInnerEfficiency_FPix[20];                // Fpix inner module efficiency
  double theOuterEfficiency_FPix[20];                // Fpix outer module efficiency
};

using namespace cms;

SiPixelDynamicInefficiencyReader::SiPixelDynamicInefficiencyReader(const edm::ParameterSet& iConfig)
    : tkGeomToken(esConsumes()),
      tkTopoToken(esConsumes()),
      dynIneffToken(esConsumes()),
      printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)) {
  //Load factors from config file, copied from SimTracker/SiPixelDigitizer/plugins/SiPixelDigitizerAlgorithm.cc
  int i = 0;
  thePixelColEfficiency[i++] = iConfig.getParameter<double>("thePixelColEfficiency_BPix1");
  thePixelColEfficiency[i++] = iConfig.getParameter<double>("thePixelColEfficiency_BPix2");
  thePixelColEfficiency[i++] = iConfig.getParameter<double>("thePixelColEfficiency_BPix3");
  i = 0;
  thePixelEfficiency[i++] = iConfig.getParameter<double>("thePixelEfficiency_BPix1");
  thePixelEfficiency[i++] = iConfig.getParameter<double>("thePixelEfficiency_BPix2");
  thePixelEfficiency[i++] = iConfig.getParameter<double>("thePixelEfficiency_BPix3");
  i = 0;
  thePixelChipEfficiency[i++] = iConfig.getParameter<double>("thePixelChipEfficiency_BPix1");
  thePixelChipEfficiency[i++] = iConfig.getParameter<double>("thePixelChipEfficiency_BPix2");
  thePixelChipEfficiency[i++] = iConfig.getParameter<double>("thePixelChipEfficiency_BPix3");
  i = 0;
  theLadderEfficiency_BPix[i++] = iConfig.getParameter<std::vector<double> >("theLadderEfficiency_BPix1");
  theLadderEfficiency_BPix[i++] = iConfig.getParameter<std::vector<double> >("theLadderEfficiency_BPix2");
  theLadderEfficiency_BPix[i++] = iConfig.getParameter<std::vector<double> >("theLadderEfficiency_BPix3");
  if ((theLadderEfficiency_BPix[0].size() != 20) || (theLadderEfficiency_BPix[1].size() != 32) ||
      (theLadderEfficiency_BPix[2].size() != 44))
    throw cms::Exception("Configuration") << "Wrong ladder number in efficiency config!";
  //
  i = 0;
  theModuleEfficiency_BPix[i++] = iConfig.getParameter<std::vector<double> >("theModuleEfficiency_BPix1");
  theModuleEfficiency_BPix[i++] = iConfig.getParameter<std::vector<double> >("theModuleEfficiency_BPix2");
  theModuleEfficiency_BPix[i++] = iConfig.getParameter<std::vector<double> >("theModuleEfficiency_BPix3");
  if ((theModuleEfficiency_BPix[0].size() != 4) || (theModuleEfficiency_BPix[1].size() != 4) ||
      (theModuleEfficiency_BPix[2].size() != 4))
    throw cms::Exception("Configuration") << "Wrong module number in efficiency config!";
  //
  i = 0;
  thePUEfficiency[i++] = iConfig.getParameter<std::vector<double> >("thePUEfficiency_BPix1");
  thePUEfficiency[i++] = iConfig.getParameter<std::vector<double> >("thePUEfficiency_BPix2");
  thePUEfficiency[i++] = iConfig.getParameter<std::vector<double> >("thePUEfficiency_BPix3");
  i = 3;
  thePixelColEfficiency[i++] = iConfig.getParameter<double>("thePixelColEfficiency_FPix1");
  thePixelColEfficiency[i++] = iConfig.getParameter<double>("thePixelColEfficiency_FPix2");
  i = 3;
  thePixelEfficiency[i++] = iConfig.getParameter<double>("thePixelEfficiency_FPix1");
  thePixelEfficiency[i++] = iConfig.getParameter<double>("thePixelEfficiency_FPix2");
  i = 3;
  thePixelChipEfficiency[i++] = iConfig.getParameter<double>("thePixelChipEfficiency_FPix1");
  thePixelChipEfficiency[i++] = iConfig.getParameter<double>("thePixelChipEfficiency_FPix2");
  //FPix Dynamic Inefficiency
  i = 3;
  theInnerEfficiency_FPix[i++] = iConfig.getParameter<double>("theInnerEfficiency_FPix1");
  theInnerEfficiency_FPix[i++] = iConfig.getParameter<double>("theInnerEfficiency_FPix2");
  i = 3;
  theOuterEfficiency_FPix[i++] = iConfig.getParameter<double>("theOuterEfficiency_FPix1");
  theOuterEfficiency_FPix[i++] = iConfig.getParameter<double>("theOuterEfficiency_FPix2");
  i = 3;
  thePUEfficiency[i++] = iConfig.getParameter<std::vector<double> >("thePUEfficiency_FPix_Inner");
  thePUEfficiency[i++] = iConfig.getParameter<std::vector<double> >("thePUEfficiency_FPix_Outer");
}

SiPixelDynamicInefficiencyReader::~SiPixelDynamicInefficiencyReader() = default;

void SiPixelDynamicInefficiencyReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const SiPixelDynamicInefficiency* SiPixelDynamicInefficiency_ = &iSetup.getData(dynIneffToken);
  edm::LogInfo("SiPixelDynamicInefficiencyReader")
      << "[SiPixelDynamicInefficiencyReader::analyze] End Reading SiPixelDynamicInefficiency" << std::endl;

  std::map<unsigned int, double> map_pixelgeomfactor = SiPixelDynamicInefficiency_->getPixelGeomFactors();
  std::map<unsigned int, double> map_colgeomfactor = SiPixelDynamicInefficiency_->getColGeomFactors();
  std::map<unsigned int, double> map_chipgeomfactor = SiPixelDynamicInefficiency_->getChipGeomFactors();
  std::map<unsigned int, std::vector<double> > map_pufactor = SiPixelDynamicInefficiency_->getPUFactors();
  std::map<unsigned int, double>::const_iterator it_pixelgeom;
  std::map<unsigned int, double>::const_iterator it_colgeom;
  std::map<unsigned int, double>::const_iterator it_chipgeom;
  std::map<unsigned int, std::vector<double> >::const_iterator it_pu;

  edm::LogPrint("SiPixelDynamicInefficiencyReader") << "Printing out DB content:" << std::endl;
  for (it_pixelgeom = map_pixelgeomfactor.begin(); it_pixelgeom != map_pixelgeomfactor.end(); it_pixelgeom++) {
    printf("pixelgeom detid %x\tfactor %f\n", it_pixelgeom->first, it_pixelgeom->second);
  }
  for (it_colgeom = map_colgeomfactor.begin(); it_colgeom != map_colgeomfactor.end(); it_colgeom++) {
    printf("colgeom detid %x\tfactor %f\n", it_colgeom->first, it_colgeom->second);
  }
  for (it_chipgeom = map_chipgeomfactor.begin(); it_chipgeom != map_chipgeomfactor.end(); it_chipgeom++) {
    printf("chipgeom detid %x\tfactor %f\n", it_chipgeom->first, it_chipgeom->second);
  }
  for (it_pu = map_pufactor.begin(); it_pu != map_pufactor.end(); it_pu++) {
    printf("pu detid %x\t", it_pu->first);
    edm::LogPrint("SiPixelDynamicInefficiencyReader") << " Size of vector " << it_pu->second.size() << " elements:";
    if (it_pu->second.size() > 1) {
      for (unsigned int i = 0; i < it_pu->second.size(); i++) {
        edm::LogPrint("SiPixelDynamicInefficiencyReader") << " " << it_pu->second.at(i);
      }
      edm::LogPrint("SiPixelDynamicInefficiencyReader") << std::endl;
    } else {
      edm::LogPrint("SiPixelDynamicInefficiencyReader") << " " << it_pu->second.at(0) << std::endl;
    }
  }
  std::vector<uint32_t> detIdmasks = SiPixelDynamicInefficiency_->getDetIdmasks();
  for (unsigned int i = 0; i < detIdmasks.size(); i++)
    printf("DetId Mask: %x\t\n", detIdmasks.at(i));
  double theInstLumiScaleFactor = SiPixelDynamicInefficiency_->gettheInstLumiScaleFactor_();
  edm::LogPrint("SiPixelDynamicInefficiencyReader") << "theInstLumiScaleFactor " << theInstLumiScaleFactor << std::endl;

  //Comparing DB factors to config factors
  edm::LogPrint("SiPixelDynamicInefficiencyReader")
      << "\nCalculating factors/module and comparing it to config file factors...\n"
      << std::endl;

  const TrackerTopology* const tTopo = &iSetup.getData(tkTopoToken);
  const TrackerGeometry* pDD = &iSetup.getData(tkGeomToken);
  edm::LogInfo("SiPixelDynamicInefficiency (old)")
      << " There are " << pDD->detUnits().size() << " detectors (old)" << std::endl;

  const size_t pu_det = map_pufactor.size();
  double _pu_scale[pu_det];
  double _pu_scale_conf[pu_det];
  unsigned int match = 0, mismatch = 0, pu_match = 0, pu_mismatch = 0;

  for (const auto& it : pDD->detUnits()) {
    if (dynamic_cast<PixelGeomDetUnit const*>(it) == nullptr)
      continue;
    const DetId detid = it->geographicalId();
    double scale_db = 1;

    //Geom DB factor calculation
    for (it_colgeom = map_colgeomfactor.begin(); it_colgeom != map_colgeomfactor.end(); ++it_colgeom) {
      const DetId mapid = DetId(it_colgeom->first);
      if (mapid.subdetId() != detid.subdetId())
        continue;
      size_t __i = 0;
      for (; __i < detIdmasks.size(); __i++) {
        DetId maskid = DetId(detIdmasks.at(__i));
        if (maskid.subdetId() != mapid.subdetId())
          continue;
        if ((detid.rawId() & maskid.rawId()) != (mapid.rawId() & maskid.rawId()) &&
            (mapid.rawId() & maskid.rawId()) != DetId(mapid.det(), mapid.subdetId()).rawId())
          break;
      }
      if (__i != detIdmasks.size())
        continue;
      scale_db *= it_colgeom->second;
    }

    //DB PU factor calculation
    it_pu = map_pufactor.begin();
    for (unsigned int pu_iterator = 0; pu_iterator < pu_det; it_pu++, pu_iterator++) {
      const DetId mapid = DetId(it_pu->first);
      if (mapid.subdetId() != detid.subdetId())
        continue;
      size_t __i = 0;
      for (; __i < detIdmasks.size(); __i++) {
        DetId maskid = DetId(detIdmasks.at(__i));
        if (maskid.subdetId() != mapid.subdetId())
          continue;
        if ((detid.rawId() & maskid.rawId()) != (mapid.rawId() & maskid.rawId()) &&
            (mapid.rawId() & maskid.rawId()) != DetId(mapid.det(), mapid.subdetId()).rawId())
          break;
      }
      if (__i != detIdmasks.size())
        continue;
      double instlumi = 30 * theInstLumiScaleFactor;
      double instlumi_pow = 1.;
      _pu_scale[pu_iterator] = 0;
      for (size_t j = 0; j < it_pu->second.size(); j++) {
        _pu_scale[pu_iterator] += instlumi_pow * it_pu->second[j];
        instlumi_pow *= instlumi;
      }
    }

    //Config PU factor calculation
    for (size_t i = 0; i < pu_det; i++) {
      double instlumi = 30 * theInstLumiScaleFactor;
      double instlumi_pow = 1.;
      _pu_scale_conf[i] = 0;
      for (size_t j = 0; j < thePUEfficiency[i].size(); j++) {
        _pu_scale_conf[i] += instlumi_pow * thePUEfficiency[i][j];
        instlumi_pow *= instlumi;
      }
    }

    //Config geom factor calculation
    double columnEfficiency = 1;
    if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
      int layerIndex = tTopo->pxbLayer(detid.rawId());
      columnEfficiency = thePixelColEfficiency[layerIndex - 1];
      int ladder = tTopo->pxbLadder(detid.rawId());
      int module = tTopo->pxbModule(detid.rawId());
      if (module <= 4)
        module = 5 - module;
      else
        module -= 4;

      columnEfficiency *=
          theLadderEfficiency_BPix[layerIndex - 1][ladder - 1] * theModuleEfficiency_BPix[layerIndex - 1][module - 1];
    }

    if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
      unsigned int diskIndex = tTopo->layer(detid) + 3;  // Use diskIndex-1 later to stay consistent with BPix
      unsigned int panelIndex = tTopo->pxfPanel(detid);
      unsigned int moduleIndex = tTopo->pxfModule(detid);
      columnEfficiency = thePixelColEfficiency[diskIndex - 1];
      if ((panelIndex == 1 && (moduleIndex == 1 || moduleIndex == 2)) ||
          (panelIndex == 2 && moduleIndex == 1)) {  //inner modules
        columnEfficiency *= theInnerEfficiency_FPix[diskIndex - 1];
      } else {  //outer modules
        columnEfficiency *= theOuterEfficiency_FPix[diskIndex - 1];
      }
    }

    if (scale_db == columnEfficiency) {
      //printf("Config match, detid %x\tfactor %f\n",detid.rawId(),columnEfficiency);
      match++;
    } else {
      //printf("Config mismatch! detid %x\t db_geom_factor %f\tconf_geom_factor %f\n",detid.rawId(),scale_db,columnEfficiency);
      mismatch++;
    }
    for (unsigned int i = 0; i < pu_det; i++) {
      if (_pu_scale[i] != 0 && _pu_scale_conf[i] != 0 && _pu_scale[i] == _pu_scale_conf[i]) {
        //printf("Config match! detid %x\t db_pu_scale %f\tconf_pu_scale %f\n",detid.rawId(),_pu_scale[i],_pu_scale_conf[i]);
        pu_match++;
        break;
      }
      if (_pu_scale[i] != 0 && _pu_scale_conf[i] != 0 && _pu_scale[i] != _pu_scale_conf[i]) {
        //printf("Config mismatch! detid %x\t db_pu_scale %f\tconf_pu_scale %f\n",detid.rawId(),_pu_scale[i],_pu_scale_conf[i]);
        pu_mismatch++;
        continue;
      }
    }
  }
  edm::LogPrint("SiPixelDynamicInefficiencyReader")
      << match << " geom factors and " << pu_match << " pu factors matched to config file factors!\n"
      << std::endl;
  if (mismatch != 0)
    edm::LogPrint("SiPixelDynamicInefficiencyReader")
        << "ERROR! " << mismatch
        << " geom factors mismatched to config file factors! Please change config and/or DB content!" << std::endl;
  if (pu_mismatch != 0)
    edm::LogPrint("SiPixelDynamicInefficiencyReader")
        << "ERROR! " << pu_mismatch
        << " pu_factors mismatched to config file pu_factors! Please change config and/or DB content!" << std::endl;
}
DEFINE_FWK_MODULE(SiPixelDynamicInefficiencyReader);

// -*- C++ -*-
//
// Package:    SiPixelCondObjAllPayloadsReader
// Class:      SiPixelCondObjAllPayloadsReader
//
/**\class SiPixelCondObjAllPayloadsReader SiPixelCondObjAllPayloadsReader.h SiPixel/test/SiPixelCondObjAllPayloadsReader.h

 Description: Test analyzer for reading pixel calibration from the DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
// $Id: SiPixelCondObjAllPayloadsReader.h,v 1.4 2009/05/28 22:12:54 dlange Exp $
//
//

// system includes
#include <memory>

// user includes
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// ROOT includes
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include <string>

namespace cms {
  class SiPixelCondObjAllPayloadsReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
  public:
    explicit SiPixelCondObjAllPayloadsReader(const edm::ParameterSet& iConfig);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override;

  private:
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
    std::unique_ptr<SiPixelGainCalibrationServiceBase> SiPixelGainCalibrationService_;

    std::map<uint32_t, TH1F*> _TH1F_Pedestals_m;
    std::map<uint32_t, TH1F*> _TH1F_Gains_m;
    TH1F* _TH1F_Gains_sum;
    TH1F* _TH1F_Pedestals_sum;
    TH1F* _TH1F_Gains_all;
    TH1F* _TH1F_Pedestals_all;
  };
}  // namespace cms

namespace cms {

  void SiPixelCondObjAllPayloadsReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("EDAnalyzer to read per-module SiPixelGainCalibration payloads in the EventSetup, for any type");
    desc.add<std::string>("payloadType", "HLT");
    descriptions.addWithDefaultLabel(desc);
  }

  SiPixelCondObjAllPayloadsReader::SiPixelCondObjAllPayloadsReader(const edm::ParameterSet& conf)
      : tkGeomToken_(esConsumes()) {
    usesResource(TFileService::kSharedResource);
    std::string payloadType = conf.getParameter<std::string>("payloadType");
    if (strcmp(payloadType.c_str(), "HLT") == 0) {
      SiPixelGainCalibrationService_ = std::make_unique<SiPixelGainCalibrationForHLTService>(conf, consumesCollector());
    } else if (strcmp(payloadType.c_str(), "Offline") == 0) {
      SiPixelGainCalibrationService_ =
          std::make_unique<SiPixelGainCalibrationOfflineService>(conf, consumesCollector());
    } else if (strcmp(payloadType.c_str(), "Full") == 0) {
      SiPixelGainCalibrationService_ = std::make_unique<SiPixelGainCalibrationService>(conf, consumesCollector());
    }
  }

  void SiPixelCondObjAllPayloadsReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    //Create Subdirectories
    edm::Service<TFileService> fs;
    TFileDirectory subDirPed = fs->mkdir("Pedestals");
    TFileDirectory subDirGain = fs->mkdir("Gains");
    char name[128];

    unsigned int nmodules = 0;
    uint32_t nchannels = 0;

    // Get the calibration data
    SiPixelGainCalibrationService_->setESObjects(iSetup);
    edm::LogInfo("SiPixelCondObjAllPayloadsReader")
        << "[SiPixelCondObjAllPayloadsReader::beginJob] End Reading CondObjects" << std::endl;

    // Get the Geometry
    const TrackerGeometry* tkgeom = &iSetup.getData(tkGeomToken_);
    edm::LogInfo("SiPixelCondObjAllPayloadsReader")
        << " There are " << tkgeom->dets().size() << " detectors" << std::endl;

    //Get list of DetIDs
    std::vector<uint32_t> vdetId_ = SiPixelGainCalibrationService_->getDetIds();

    //Create histograms
    _TH1F_Gains_sum = fs->make<TH1F>("Summary_Gain", "Gain Summary", vdetId_.size() + 1, 0, vdetId_.size() + 1);
    _TH1F_Pedestals_sum =
        fs->make<TH1F>("Summary_Pedestal", "Pedestal Summary", vdetId_.size() + 1, 0, vdetId_.size() + 1);
    _TH1F_Pedestals_all = fs->make<TH1F>("PedestalsAll", "all Pedestals", 350, -100, 250);
    _TH1F_Gains_all = fs->make<TH1F>("GainsAll", "all Gains", 100, 0, 10);

    // Loop over DetId's
    int ibin = 1;
    for (std::vector<uint32_t>::const_iterator detid_iter = vdetId_.begin(); detid_iter != vdetId_.end();
         detid_iter++) {
      uint32_t detid = *detid_iter;

      sprintf(name, "Pedestals_%d", detid);
      _TH1F_Pedestals_m[detid] = subDirPed.make<TH1F>(name, name, 250, 0., 250.);
      sprintf(name, "Gains_%d", detid);
      _TH1F_Gains_m[detid] = subDirGain.make<TH1F>(name, name, 100, 0., 10.);

      DetId detIdObject(detid);
      const PixelGeomDetUnit* _PixelGeomDetUnit =
          dynamic_cast<const PixelGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
      if (_PixelGeomDetUnit == nullptr) {
        edm::LogError("SiPixelCondObjDisplay") << "[SiPixelCondObjAllPayloadsReader::beginJob] the detID " << detid
                                               << " doesn't seem to belong to Tracker" << std::endl;
        continue;
      }

      nmodules++;

      const GeomDetUnit* geoUnit = tkgeom->idToDetUnit(detIdObject);
      const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
      const PixelTopology& topol = pixDet->specificTopology();

      // Get the module sizes.
      int nrows = topol.nrows();     // rows in x
      int ncols = topol.ncolumns();  // cols in y

      for (int col_iter = 0; col_iter < ncols; col_iter++) {
        for (int row_iter = 0; row_iter < nrows; row_iter++) {
          nchannels++;

          float gain = SiPixelGainCalibrationService_->getGain(detid, col_iter, row_iter);
          _TH1F_Gains_m[detid]->Fill(gain);
          _TH1F_Gains_all->Fill(gain);

          float ped = SiPixelGainCalibrationService_->getPedestal(detid, col_iter, row_iter);
          _TH1F_Pedestals_m[detid]->Fill(ped);
          _TH1F_Pedestals_all->Fill(ped);

          //edm::LogPrint("SiPixelCondObjAllPayloadsReader") << "       Col "<<col_iter<<" Row "<<row_iter<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
        }
      }

      _TH1F_Gains_sum->SetBinContent(ibin, _TH1F_Gains_m[detid]->GetMean());
      _TH1F_Gains_sum->SetBinError(ibin, _TH1F_Gains_m[detid]->GetRMS());
      _TH1F_Pedestals_sum->SetBinContent(ibin, _TH1F_Pedestals_m[detid]->GetMean());
      _TH1F_Pedestals_sum->SetBinError(ibin, _TH1F_Pedestals_m[detid]->GetRMS());

      ibin++;
    }

    edm::LogInfo("SiPixelCondObjAllPayloadsReader")
        << "[SiPixelCondObjAllPayloadsReader::analyze] ---> PIXEL Modules  " << nmodules << std::endl;
    edm::LogInfo("SiPixelCondObjAllPayloadsReader")
        << "[SiPixelCondObjAllPayloadsReader::analyze] ---> PIXEL Channels " << nchannels << std::endl;
  }

  // ------------ method called once each job just after ending the event loop  ------------
  void SiPixelCondObjAllPayloadsReader::endJob() {
    edm::LogPrint("SiPixelCondObjAllPayloadsReader") << " ---> End job " << std::endl;
  }
}  // namespace cms

using namespace cms;
DEFINE_FWK_MODULE(SiPixelCondObjAllPayloadsReader);

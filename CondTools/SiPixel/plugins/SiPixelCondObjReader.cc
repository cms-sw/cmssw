// -*- C++ -*-
//
// Package:    SiPixelCondObjReader
// Class:      SiPixelCondObjReader
//
/**\class SiPixelCondObjReader SiPixelCondObjReader.h SiPixel/plugins/SiPixelCondObjReader.cc

 Description: Test analyzer for reading pixel calibration from the DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
// $Id: SiPixelCondObjReader.h,v 1.8 2009/05/28 22:12:55 dlange Exp $
//
//

// system includes
#include <memory>

// user includes
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationService.h"

// ROOT includes
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"

namespace cms {
  class SiPixelCondObjReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
  public:
    explicit SiPixelCondObjReader(const edm::ParameterSet &iConfig);

    ~SiPixelCondObjReader() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
    void analyze(const edm::Event &, const edm::EventSetup &) override;
    void endJob() override;

  private:
    edm::ParameterSet conf_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
    SiPixelGainCalibrationService SiPixelGainCalibrationService_;

    std::map<uint32_t, TH1F *> _TH1F_Pedestals_m;
    std::map<uint32_t, TH1F *> _TH1F_Gains_m;
    std::map<uint32_t, double> _deadfrac_m;
    std::map<uint32_t, double> _noisyfrac_m;

    TH1F *_TH1F_Dead_sum;
    TH1F *_TH1F_Noisy_sum;
    TH1F *_TH1F_Gains_sum;
    TH1F *_TH1F_Pedestals_sum;
    TH1F *_TH1F_Dead_all;
    TH1F *_TH1F_Noisy_all;
    TH1F *_TH1F_Gains_all;
    TH1F *_TH1F_Pedestals_all;
    TH1F *_TH1F_Gains_bpix;
    TH1F *_TH1F_Gains_fpix;
    TH1F *_TH1F_Pedestals_bpix;
    TH1F *_TH1F_Pedestals_fpix;
  };
}  // namespace cms

namespace cms {
  SiPixelCondObjReader::SiPixelCondObjReader(const edm::ParameterSet &conf)
      : conf_(conf), tkGeomToken_(esConsumes()), SiPixelGainCalibrationService_(conf, consumesCollector()) {
    usesResource(TFileService::kSharedResource);
  }

  void SiPixelCondObjReader::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
    //Create Subdirectories
    edm::Service<TFileService> fs;
    TFileDirectory subDirPed = fs->mkdir("Pedestals");
    TFileDirectory subDirGain = fs->mkdir("Gains");
    char name[128];

    unsigned int nmodules = 0;
    uint32_t nchannels = 0;
    uint32_t ndead = 0;
    uint32_t nnoisy = 0;

    // Get the calibration data
    SiPixelGainCalibrationService_.setESObjects(iSetup);
    edm::LogInfo("SiPixelCondObjReader") << "[SiPixelCondObjReader::beginJob] End Reading CondObjects" << std::endl;

    // Get the Geometry
    const TrackerGeometry *tkgeom = &iSetup.getData(tkGeomToken_);
    edm::LogInfo("SiPixelCondObjReader") << " There are " << tkgeom->dets().size() << " detectors" << std::endl;

    // Get the list of DetId's
    std::vector<uint32_t> vdetId_ = SiPixelGainCalibrationService_.getDetIds();

    //Create histograms
    _TH1F_Dead_sum = fs->make<TH1F>(
        "Summary_dead", "Dead pixel fraction (0=dead, 1=alive)", vdetId_.size() + 1, 0, vdetId_.size() + 1);
    _TH1F_Dead_all = fs->make<TH1F>("DeadAll",
                                    "Dead pixel fraction (0=dead, 1=alive)",
                                    50,
                                    0.,
                                    conf_.getUntrackedParameter<double>("maxRangeDeadPixHist", 0.001));
    _TH1F_Noisy_sum = fs->make<TH1F>(
        "Summary_noisy", "Noisy pixel fraction (0=noisy, 1=alive)", vdetId_.size() + 1, 0, vdetId_.size() + 1);
    _TH1F_Noisy_all = fs->make<TH1F>("NoisyAll",
                                     "Noisy pixel fraction (0=noisy, 1=alive)",
                                     50,
                                     0.,
                                     conf_.getUntrackedParameter<double>("maxRangeDeadPixHist", 0.001));
    _TH1F_Gains_sum = fs->make<TH1F>("Summary_Gain", "Gain Summary", vdetId_.size() + 1, 0, vdetId_.size() + 1);
    _TH1F_Pedestals_sum =
        fs->make<TH1F>("Summary_Pedestal", "Pedestal Summary", vdetId_.size() + 1, 0, vdetId_.size() + 1);
    _TH1F_Pedestals_all = fs->make<TH1F>("PedestalsAll", "all Pedestals", 350, -100, 250);
    _TH1F_Pedestals_bpix = fs->make<TH1F>("PedestalsBpix", "bpix Pedestals", 350, -100, 250);
    _TH1F_Pedestals_fpix = fs->make<TH1F>("PedestalsFpix", "fpix Pedestals", 350, -100, 250);
    _TH1F_Gains_all = fs->make<TH1F>("GainsAll", "all Gains", 100, 0, 10);
    _TH1F_Gains_bpix = fs->make<TH1F>("GainsBpix", "bpix Gains", 100, 0, 10);
    _TH1F_Gains_fpix = fs->make<TH1F>("GainsFpix", "fpix Gains", 100, 0, 10);

    TTree *tree = new TTree("tree", "tree");
    uint32_t detid;
    double gainmeanfortree, gainrmsfortree, pedmeanfortree, pedrmsfortree;
    tree->Branch("detid", &detid, "detid/I");
    tree->Branch("ped_mean", &pedmeanfortree, "ped_mean/D");
    tree->Branch("ped_rms", &pedrmsfortree, "ped_rms/D");
    tree->Branch("gain_mean", &gainmeanfortree, "gain_mean/D");
    tree->Branch("gain_rms", &gainrmsfortree, "gain_rms/D");

    // Loop over DetId's
    int ibin = 1;
    for (std::vector<uint32_t>::const_iterator detid_iter = vdetId_.begin(); detid_iter != vdetId_.end();
         detid_iter++) {
      detid = *detid_iter;

      sprintf(name, "Pedestals_%d", detid);
      _TH1F_Pedestals_m[detid] = subDirPed.make<TH1F>(name, name, 350, -100., 250.);
      sprintf(name, "Gains_%d", detid);
      _TH1F_Gains_m[detid] = subDirGain.make<TH1F>(name, name, 100, 0., 10.);

      DetId detIdObject(detid);
      const PixelGeomDetUnit *_PixelGeomDetUnit =
          dynamic_cast<const PixelGeomDetUnit *>(tkgeom->idToDetUnit(DetId(detid)));
      if (_PixelGeomDetUnit == nullptr) {
        edm::LogError("SiPixelCondObjDisplay") << "[SiPixelCondObjReader::beginJob] the detID " << detid
                                               << " doesn't seem to belong to Tracker" << std::endl;
        continue;
      }

      _deadfrac_m[detid] = 0.;
      _noisyfrac_m[detid] = 0.;

      nmodules++;

      const GeomDetUnit *geoUnit = tkgeom->idToDetUnit(detIdObject);
      const PixelGeomDetUnit *pixDet = dynamic_cast<const PixelGeomDetUnit *>(geoUnit);
      const PixelTopology &topol = pixDet->specificTopology();

      // Get the module sizes.
      int nrows = topol.nrows();     // rows in x
      int ncols = topol.ncolumns();  // cols in y
      float nchannelspermod = 0;

      for (int col_iter = 0; col_iter < ncols; col_iter++) {
        for (int row_iter = 0; row_iter < nrows; row_iter++) {
          nchannelspermod++;
          nchannels++;

          if (SiPixelGainCalibrationService_.isDead(detid, col_iter, row_iter)) {
            //	    edm::LogPrint("SiPixelCondObjReader") << "found dead pixel " << detid << " " <<col_iter << "," << row_iter << std::endl;
            ndead++;
            _deadfrac_m[detid]++;
            continue;
          } else if (SiPixelGainCalibrationService_.isNoisy(detid, col_iter, row_iter)) {
            //	    edm::LogPrint("SiPixelCondObjReader") << "found noisy pixel " << detid << " " <<col_iter << "," << row_iter << std::endl;
            nnoisy++;
            _noisyfrac_m[detid]++;
            continue;
          }

          float gain = SiPixelGainCalibrationService_.getGain(detid, col_iter, row_iter);
          _TH1F_Gains_m[detid]->Fill(gain);
          _TH1F_Gains_all->Fill(gain);

          if (detIdObject.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel))
            _TH1F_Gains_bpix->Fill(gain);
          if (detIdObject.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap))
            _TH1F_Gains_fpix->Fill(gain);

          float ped = SiPixelGainCalibrationService_.getPedestal(detid, col_iter, row_iter);
          _TH1F_Pedestals_m[detid]->Fill(ped);
          _TH1F_Pedestals_all->Fill(ped);
          //	 edm::LogPrint("SiPixelCondObjReader")<<"detid  "<<detid<<"     ped "<<ped<<std::endl;

          if (detIdObject.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel))
            _TH1F_Pedestals_bpix->Fill(ped);
          if (detIdObject.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap))
            _TH1F_Pedestals_fpix->Fill(ped);

          //	 edm::LogPrint("SiPixelCondObjReader") <<"    DetId "<<detid<<"       Col "<<col_iter<<" Row "<<row_iter<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
        }
      }

      _deadfrac_m[detid] /= nchannelspermod;
      _noisyfrac_m[detid] /= nchannelspermod;
      _TH1F_Dead_sum->SetBinContent(ibin, _deadfrac_m[detid]);
      _TH1F_Dead_all->Fill(_deadfrac_m[detid]);
      _TH1F_Noisy_sum->SetBinContent(ibin, _noisyfrac_m[detid]);
      _TH1F_Noisy_all->Fill(_noisyfrac_m[detid]);
      _TH1F_Gains_sum->SetBinContent(ibin, _TH1F_Gains_m[detid]->GetMean());
      _TH1F_Gains_sum->SetBinError(ibin, _TH1F_Gains_m[detid]->GetRMS());
      _TH1F_Pedestals_sum->SetBinContent(ibin, _TH1F_Pedestals_m[detid]->GetMean());
      _TH1F_Pedestals_sum->SetBinError(ibin, _TH1F_Pedestals_m[detid]->GetRMS());

      gainmeanfortree = _TH1F_Gains_m[detid]->GetMean();
      gainrmsfortree = _TH1F_Gains_m[detid]->GetRMS();
      pedmeanfortree = _TH1F_Pedestals_m[detid]->GetMean();
      pedrmsfortree = _TH1F_Pedestals_m[detid]->GetRMS();
      edm::LogPrint("SiPixelCondObjReader")
          << "DetId " << detid << "       GainMean " << gainmeanfortree << " RMS " << gainrmsfortree << "      PedMean "
          << pedmeanfortree << " RMS " << pedrmsfortree << std::endl;
      tree->Fill();

      ibin++;
    }

    edm::LogInfo("SiPixelCondObjReader") << "[SiPixelCondObjReader::analyze] ---> PIXEL Modules  " << nmodules
                                         << std::endl;
    edm::LogInfo("SiPixelCondObjReader") << "[SiPixelCondObjReader::analyze] ---> PIXEL Channels " << nchannels
                                         << std::endl;

    edm::LogPrint("SiPixelCondObjReader") << " ---> SUMMARY :" << std::endl;
    edm::LogPrint("SiPixelCondObjReader") << "Encounted " << ndead << " dead pixels" << std::endl;
    edm::LogPrint("SiPixelCondObjReader") << "Encounted " << nnoisy << " noisy pixels" << std::endl;
    edm::LogPrint("SiPixelCondObjReader")
        << "The Gain Mean is " << _TH1F_Gains_all->GetMean() << " with rms " << _TH1F_Gains_all->GetRMS() << std::endl;
    edm::LogPrint("SiPixelCondObjReader") << "         in BPIX " << _TH1F_Gains_bpix->GetMean() << " with rms "
                                          << _TH1F_Gains_bpix->GetRMS() << std::endl;
    edm::LogPrint("SiPixelCondObjReader") << "         in FPIX " << _TH1F_Gains_fpix->GetMean() << " with rms "
                                          << _TH1F_Gains_fpix->GetRMS() << std::endl;
    edm::LogPrint("SiPixelCondObjReader") << "The Ped Mean is " << _TH1F_Pedestals_all->GetMean() << " with rms "
                                          << _TH1F_Pedestals_all->GetRMS() << std::endl;
    edm::LogPrint("SiPixelCondObjReader") << "         in BPIX " << _TH1F_Pedestals_bpix->GetMean() << " with rms "
                                          << _TH1F_Pedestals_bpix->GetRMS() << std::endl;
    edm::LogPrint("SiPixelCondObjReader") << "         in FPIX " << _TH1F_Pedestals_fpix->GetMean() << " with rms "
                                          << _TH1F_Pedestals_fpix->GetRMS() << std::endl;
  }

  // ------------ method called once each job just after ending the event loop  ------------
  void SiPixelCondObjReader::endJob() { edm::LogPrint("SiPixelCondObjReader") << " ---> End job " << std::endl; }

  void SiPixelCondObjReader::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("EDAnalyzer to read per-module SiPixelGainCalibration payloads in the EventSetup");
    desc.addUntracked<double>("maxRangeDeadPixHist", 0.001);
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace cms
using namespace cms;
DEFINE_FWK_MODULE(SiPixelCondObjReader);

#include <memory>

#include "CalibTracker/SiPixelESProducers/test/SiPixelFakeGainOfflineReader.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
namespace cms {
  SiPixelFakeGainOfflineReader::SiPixelFakeGainOfflineReader(const edm::ParameterSet& conf)
      : conf_(conf),
        SiPixelGainCalibrationOfflineService_(conf),
        filename_(conf.getParameter<std::string>("fileName")) {}

  void SiPixelFakeGainOfflineReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    unsigned int nmodules = 0;
    uint32_t nchannels = 0;
    fFile->cd();

    // Get the Geometry
    iSetup.get<TrackerDigiGeometryRecord>().get(tkgeom);
    edm::LogInfo("SiPixelFakeGainOfflineReader") << " There are " << tkgeom->dets().size() << " detectors" << std::endl;

    // Get the calibrationOffline data
    //iSetup.get<SiPixelGainCalibrationOfflineRcd>().get(SiPixelGainCalibrationOffline_);
    //edm::LogInfo("SiPixelFakeGainOfflineReader") << "[SiPixelFakeGainOfflineReader::analyze] End Reading FakeGainOfflineects" << std::endl;
    //SiPixelGainCalibrationOfflineService_.setESObjects(iSetup);

    //  for(TrackerGeometry::DetContainer::const_iterator it = tkgeom->dets().begin(); it != tkgeom->dets().end(); it++){
    //   if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
    //     uint32_t detid=((*it)->geographicalId()).rawId();
    // Get the list of DetId's

    std::vector<uint32_t> vdetId_ = SiPixelGainCalibrationOfflineService_.getDetIds();
    // Loop over DetId's
    for (std::vector<uint32_t>::const_iterator detid_iter = vdetId_.begin(); detid_iter != vdetId_.end();
         detid_iter++) {
      uint32_t detid = *detid_iter;

      DetId detIdObject(detid);
      nmodules++;
      //if(nmodules>3) break;

      std::map<uint32_t, TH1F*>::iterator p_iter = _TH1F_Pedestals_m.find(detid);
      std::map<uint32_t, TH1F*>::iterator g_iter = _TH1F_Gains_m.find(detid);

      const GeomDetUnit* geoUnit = tkgeom->idToDetUnit(detIdObject);
      const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
      const PixelTopology& topol = pixDet->specificTopology();

      // Get the module sizes.
      int nrows = topol.nrows();     // rows in x
      int ncols = topol.ncolumns();  // cols in y

      for (int col_iter = 0; col_iter < ncols; col_iter++) {
        for (int row_iter = 0; row_iter < nrows; row_iter++) {
          nchannels++;

          float gain = SiPixelGainCalibrationOfflineService_.getGain(detid, col_iter, row_iter);
          g_iter->second->Fill(gain);
          float ped = SiPixelGainCalibrationOfflineService_.getPedestal(detid, col_iter, row_iter);
          p_iter->second->Fill(ped);

          //std::cout << "       Col "<<col_iter<<" Row "<<row_iter<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
        }
      }
    }

    edm::LogInfo("SiPixelFakeGainOfflineReader")
        << "[SiPixelFakeGainOfflineReader::analyze] ---> PIXEL Modules  " << nmodules << std::endl;
    edm::LogInfo("SiPixelFakeGainOfflineReader")
        << "[SiPixelFakeGainOfflineReader::analyze] ---> PIXEL Channels " << nchannels << std::endl;
    fFile->ls();
    fFile->Write();
    fFile->Close();
  }

  // ------------ method called once each job just before starting event loop  ------------
  void SiPixelFakeGainOfflineReader::beginJob() {}

  // ------------ method called once each job just before starting event loop  ------------
  void SiPixelFakeGainOfflineReader::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
    static int first(1);
    if (1 == first) {
      first = 0;
      edm::LogInfo("SiPixelFakeGainOfflineReader")
          << "[SiPixelFakeGainOfflineReader::beginJob] Opening ROOT file  " << std::endl;
      fFile = new TFile(filename_.c_str(), "RECREATE");
      fFile->mkdir("Pedestals");
      fFile->mkdir("Gains");
      fFile->cd();
      char name[128];

      // Get Geometry
      iSetup.get<TrackerDigiGeometryRecord>().get(tkgeom);

      // Get the calibrationOffline data
      //edm::ESHandle<SiPixelGainCalibrationOffline> SiPixelGainCalibrationOffline_;
      //iSetup.get<SiPixelGainCalibrationOfflineRcd>().get(SiPixelGainCalibrationOffline_);
      SiPixelGainCalibrationOfflineService_.setESObjects(iSetup);
      edm::LogInfo("SiPixelFakeGainOfflineReader")
          << "[SiPixelFakeGainOfflineReader::beginJob] End Reading FakeGainOfflineects" << std::endl;
      // Get the list of DetId's
      std::vector<uint32_t> vdetId_ = SiPixelGainCalibrationOfflineService_.getDetIds();
      //SiPixelGainCalibrationOffline_->getDetIds(vdetId_);
      // Loop over DetId's
      for (std::vector<uint32_t>::const_iterator detid_iter = vdetId_.begin(); detid_iter != vdetId_.end();
           detid_iter++) {
        uint32_t detid = *detid_iter;

        const PixelGeomDetUnit* _PixelGeomDetUnit =
            dynamic_cast<const PixelGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
        if (_PixelGeomDetUnit == 0) {
          edm::LogError("SiPixelFakeGainOfflineDisplay") << "[SiPixelFakeGainOfflineReader::beginJob] the detID "
                                                         << detid << " doesn't seem to belong to Tracker" << std::endl;
          continue;
        }
        // Book histograms
        sprintf(name, "Pedestals_%d", detid);
        fFile->cd();
        fFile->cd("Pedestals");
        _TH1F_Pedestals_m[detid] = new TH1F(name, name, 50, 0., 50.);
        sprintf(name, "Gains_%d", detid);
        fFile->cd();
        fFile->cd("Gains");
        _TH1F_Gains_m[detid] = new TH1F(name, name, 100, 0., 10.);
      }
    }
  }

  // ------------ method called once each job just after ending the event loop  ------------
  void SiPixelFakeGainOfflineReader::endJob() { std::cout << " ---> End job " << std::endl; }
}  // namespace cms

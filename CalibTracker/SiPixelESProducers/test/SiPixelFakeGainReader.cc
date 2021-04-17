#include <memory>

#include "CalibTracker/SiPixelESProducers/test/SiPixelFakeGainReader.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
namespace cms {
  SiPixelFakeGainReader::SiPixelFakeGainReader(const edm::ParameterSet& conf)
      : conf_(conf),
        trackerGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
        trackerGeomTokenBeginRun_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
        SiPixelGainCalibrationService_(conf),
        filename_(conf.getParameter<std::string>("fileName")) {}

  void SiPixelFakeGainReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    unsigned int nmodules = 0;
    uint32_t nchannels = 0;
    fFile->cd();

    // Get the Geometry
    edm::ESHandle<TrackerGeometry> tkgeom = iSetup.getHandle(trackerGeomToken_);
    edm::LogInfo("SiPixelFakeGainReader") << " There are " << tkgeom->dets().size() << " detectors" << std::endl;

    //  for(TrackerGeometry::DetContainer::const_iterator it = tkgeom->dets().begin(); it != tkgeom->dets().end(); it++){
    //   if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
    //     uint32_t detid=((*it)->geographicalId()).rawId();
    // Get the list of DetId's

    std::vector<uint32_t> vdetId_ = SiPixelGainCalibrationService_.getDetIds();
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

          float ped = SiPixelGainCalibrationService_.getPedestal(detid, col_iter, row_iter);
          p_iter->second->Fill(ped);

          float gain = SiPixelGainCalibrationService_.getGain(detid, col_iter, row_iter);
          g_iter->second->Fill(gain);

          //std::cout << "       Col "<<col_iter<<" Row "<<row_iter<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
        }
      }
    }

    edm::LogInfo("SiPixelFakeGainReader")
        << "[SiPixelFakeGainReader::analyze] ---> PIXEL Modules  " << nmodules << std::endl;
    edm::LogInfo("SiPixelFakeGainReader")
        << "[SiPixelFakeGainReader::analyze] ---> PIXEL Channels " << nchannels << std::endl;
    fFile->ls();
    fFile->Write();
    fFile->Close();
  }

  // ------------ method called once each job just before starting event loop  ------------
  void SiPixelFakeGainReader::beginJob() {}

  // ------------ method called once each job just before starting event loop  ------------
  void SiPixelFakeGainReader::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
    static int first(1);
    if (1 == first) {
      first = 0;
      edm::LogInfo("SiPixelFakeGainReader") << "[SiPixelFakeGainReader::beginJob] Opening ROOT file  " << std::endl;
      fFile = new TFile(filename_.c_str(), "RECREATE");
      fFile->mkdir("Pedestals");
      fFile->mkdir("Gains");
      fFile->cd();
      char name[128];

      // Get Geometry
      edm::ESHandle<TrackerGeometry> tkgeom = iSetup.getHandle(trackerGeomTokenBeginRun_);

      // Get the calibration data
      SiPixelGainCalibrationService_.setESObjects(iSetup);
      edm::LogInfo("SiPixelFakeGainReader")
          << "[SiPixelFakeGainReader::beginJob] End Reading FakeGainects" << std::endl;
      // Get the list of DetId's
      std::vector<uint32_t> vdetId_ = SiPixelGainCalibrationService_.getDetIds();
      // Loop over DetId's
      for (std::vector<uint32_t>::const_iterator detid_iter = vdetId_.begin(); detid_iter != vdetId_.end();
           detid_iter++) {
        uint32_t detid = *detid_iter;

        const PixelGeomDetUnit* _PixelGeomDetUnit =
            dynamic_cast<const PixelGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
        if (_PixelGeomDetUnit == 0) {
          edm::LogError("SiPixelFakeGainDisplay") << "[SiPixelFakeGainReader::beginJob] the detID " << detid
                                                  << " doesn't seem to belong to Tracker" << std::endl;
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
  void SiPixelFakeGainReader::endJob() { std::cout << " ---> End job " << std::endl; }
}  // namespace cms

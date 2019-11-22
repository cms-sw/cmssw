#include <memory>
#include <iostream>
#include "CondTools/SiPixel/test/SiPixelCondObjOfflineBuilder.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"

namespace cms {
  SiPixelCondObjOfflineBuilder::SiPixelCondObjOfflineBuilder(const edm::ParameterSet& iConfig)
      : conf_(iConfig),
        appendMode_(conf_.getUntrackedParameter<bool>("appendMode", true)),
        SiPixelGainCalibration_(0),
        SiPixelGainCalibrationService_(iConfig),
        recordName_(iConfig.getParameter<std::string>("record")),
        meanPed_(conf_.getParameter<double>("meanPed")),
        rmsPed_(conf_.getParameter<double>("rmsPed")),
        meanGain_(conf_.getParameter<double>("meanGain")),
        rmsGain_(conf_.getParameter<double>("rmsGain")),
        meanPedFPix_(conf_.getUntrackedParameter<double>("meanPedFPix", meanPed_)),
        rmsPedFPix_(conf_.getUntrackedParameter<double>("rmsPedFPix", rmsPed_)),
        meanGainFPix_(conf_.getUntrackedParameter<double>("meanGainFPix", meanGain_)),
        rmsGainFPix_(conf_.getUntrackedParameter<double>("rmsGainFPix", rmsGain_)),
        deadFraction_(conf_.getParameter<double>("deadFraction")),
        noisyFraction_(conf_.getParameter<double>("noisyFraction")),
        secondRocRowGainOffset_(conf_.getParameter<double>("secondRocRowGainOffset")),
        secondRocRowPedOffset_(conf_.getParameter<double>("secondRocRowPedOffset")),
        numberOfModules_(conf_.getParameter<int>("numberOfModules")),
        fromFile_(conf_.getParameter<bool>("fromFile")),
        fileName_(conf_.getParameter<std::string>("fileName")),
        generateColumns_(conf_.getUntrackedParameter<bool>("generateColumns", true)) {
    ::putenv((char*)"CORAL_AUTH_USER=me");
    ::putenv((char*)"CORAL_AUTH_PASSWORD=test");
  }

  void SiPixelCondObjOfflineBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm;
    unsigned int run = iEvent.id().run();
    int nmodules = 0;
    uint32_t nchannels = 0;
    //    int mycol = 415;
    //    int myrow = 159;

    edm::LogInfo("SiPixelCondObjOfflineBuilder")
        << "... creating dummy SiPixelGainCalibration Data for Run " << run << "\n " << std::endl;
    //
    // Instantiate Gain calibration offset and define pedestal/gain range
    //
    // note: the hard-coded range values are also used in random generation. That is why they're defined here

    float minped = 0.;
    float maxped = 100.;
    float mingain = 0.;
    float maxgain = 10.;
    SiPixelGainCalibration_ = new SiPixelGainCalibrationOffline(minped, maxped, mingain, maxgain);

    edm::ESHandle<TrackerGeometry> pDD;
    iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
    edm::LogInfo("SiPixelCondObjOfflineBuilder") << " There are " << pDD->dets().size() << " detectors" << std::endl;

    for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
      if (dynamic_cast<PixelGeomDetUnit const*>((*it)) != 0) {
        uint32_t detid = ((*it)->geographicalId()).rawId();

        // Stop if module limit reached
        nmodules++;
        if (nmodules > numberOfModules_)
          break;

        const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>((*it));
        const PixelTopology& topol = pixDet->specificTopology();
        // Get the module sizes.
        int nrows = topol.nrows();     // rows in x
        int ncols = topol.ncolumns();  // cols in y
        //std::cout << " ---> PIXEL DETID " << detid << " Cols " << ncols << " Rows " << nrows << std::endl;

        double meanPedWork = meanPed_;
        double rmsPedWork = rmsPed_;
        double meanGainWork = meanGain_;
        double rmsGainWork = rmsGain_;
        DetId detId(detid);
        if (detId.subdetId() == 2) {  // FPIX
          meanPedWork = meanPedFPix_;
          rmsPedWork = rmsPedFPix_;
          meanGainWork = meanGainFPix_;
          rmsGainWork = rmsGainFPix_;
        }

        PixelIndices pIndexConverter(ncols, nrows);

        std::vector<char> theSiPixelGainCalibration;

        // Loop over columns and rows of this DetID
        for (int i = 0; i < ncols; i++) {
          float totalGain = 0.0;
          for (int j = 0; j < nrows; j++) {
            nchannels++;
            bool isDead = false;
            bool isNoisy = false;
            float ped = 0.0, gain = 0.0;

            if (fromFile_) {
              // Use calibration from a file
              int chipIndex = 0, colROC = 0, rowROC = 0;

              pIndexConverter.transformToROC(i, j, chipIndex, colROC, rowROC);
              int chanROC = PixelIndices::pixelToChannelROC(rowROC, colROC);  // use ROC coordinates
              //	     float pp0=0, pp1=0;
              std::map<int, CalParameters, std::less<int> >::const_iterator it = calmap_.find(chanROC);
              CalParameters theCalParameters = (*it).second;
              ped = theCalParameters.p0;
              gain = theCalParameters.p1;

            } else {
              if (deadFraction_ > 0) {
                double val = CLHEP::RandFlat::shoot();
                if (val < deadFraction_) {
                  isDead = true;
                  //		 std::cout << "dead pixel " << detid << " " << i << "," << j << " " << val << std::endl;
                }
              }
              if (deadFraction_ > 0 && !isDead) {
                double val = CLHEP::RandFlat::shoot();
                if (val < noisyFraction_) {
                  isNoisy = true;
                  //		 std::cout << "noisy pixel " << detid << " " << i << "," << j << " " << val << std::endl;
                }
              }

              if (rmsPedWork > 0) {
                ped = CLHEP::RandGauss::shoot(meanPedWork, rmsPedWork);
                while (ped < minped || ped > maxped)
                  ped = CLHEP::RandGauss::shoot(meanPedWork, rmsPedWork);
              } else
                ped = meanPedWork;
              if (rmsGainWork > 0) {
                gain = CLHEP::RandGauss::shoot(meanGainWork, rmsGainWork);
                while (gain < mingain || gain > maxgain)
                  gain = CLHEP::RandGauss::shoot(meanGainWork, rmsGainWork);

              } else
                gain = meanGainWork;
            }

            // 	   if(i==mycol && j==myrow) {
            //	     std::cout << "       Col "<<i<<" Row "<<j<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
            // 	   }

            //  	   gain =  2.8;
            //  	   ped  = 28.2;

            //if in the second row of rocs (i.e. a 2xN plaquette) add an offset (if desired) for testing
            if (j >= 80) {
              ped += secondRocRowPedOffset_;
              gain += secondRocRowGainOffset_;

              if (gain > maxgain)
                gain = maxgain;
              else if (gain < mingain)
                gain = mingain;

              if (ped > maxped)
                ped = maxped;
              else if (ped < minped)
                ped = minped;
            }

            totalGain += gain;

            if (!isDead && !isNoisy) {
              SiPixelGainCalibration_->setDataPedestal(ped, theSiPixelGainCalibration);
            } else if (isDead)  // dead pixel
              //	     std::cout << "filling pixel as dead for detid " << detid <<", col " << i << ", row" << j <<  std::endl;
              SiPixelGainCalibration_->setDeadPixel(theSiPixelGainCalibration);
            else if (isNoisy)  // dead pixel
              //	     std::cout << "filling pixel as dead for detid " << detid <<", col " << i << ", row" << j <<  std::endl;
              SiPixelGainCalibration_->setNoisyPixel(theSiPixelGainCalibration);
            if ((j + 1) % 80 == 0)  // fill the column average after ever ROC!
            {
              float averageGain = totalGain / static_cast<float>(80);

              if (generateColumns_) {
                averageGain = gain;
              }

              //std::cout << "Filling gain " << averageGain << " for col: " << i << " row: " << j << std::endl;
              SiPixelGainCalibration_->setDataGain(averageGain, 80, theSiPixelGainCalibration);
              totalGain = 0;
            }
          }
        }

        SiPixelGainCalibrationOffline::Range range(theSiPixelGainCalibration.begin(), theSiPixelGainCalibration.end());
        if (!SiPixelGainCalibration_->put(detid, range, ncols))
          edm::LogError("SiPixelCondObjOfflineBuilder")
              << "[SiPixelCondObjOfflineBuilder::analyze] detid already exists" << std::endl;
      }
    }
    std::cout << " ---> PIXEL Modules  " << nmodules << std::endl;
    std::cout << " ---> PIXEL Channels " << nchannels << std::endl;

    //   // Try to read object
    //    int mynmodules =0;
    //    for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    //      if( dynamic_cast<PixelGeomDetUnit const*>((*it))!=0){
    //        uint32_t mydetid=((*it)->geographicalId()).rawId();
    //        mynmodules++;
    //        if( mynmodules > numberOfModules_) break;
    //        SiPixelGainCalibration::Range myrange = SiPixelGainCalibration_->getRange(mydetid);
    //        float mypedestal = SiPixelGainCalibration_->getPed (mycol,myrow,myrange,416);
    //        float mygain     = SiPixelGainCalibration_->getGain(mycol,myrow,myrange,416);
    //        //std::cout<<" PEDESTAL "<< mypedestal<<" GAIN "<<mygain<<std::endl;
    //      }
    //    }
    // Write into DB
    edm::LogInfo("SiPixelCondOfflineBuilder") << "writing to DB, record = \"" << recordName_ << "\"";
    edm::Service<cond::service::PoolDBOutputService> mydbservice;
    if (!mydbservice.isAvailable()) {
      edm::LogError("SiPixelCondOfflineBuilder") << " db service unavailable";
      return;
    } else {
      edm::LogInfo("SiPixelCondOfflineBuilder") << " DB service OK";
    }

    try {
      if (mydbservice->isNewTagRequest(recordName_.c_str())) {
        mydbservice->createNewIOV<SiPixelGainCalibrationOffline>(
            SiPixelGainCalibration_, mydbservice->beginOfTime(), mydbservice->endOfTime(), recordName_.c_str());
      } else {
        mydbservice->appendSinceTime<SiPixelGainCalibrationOffline>(
            SiPixelGainCalibration_, mydbservice->currentTime(), recordName_.c_str());
      }
      edm::LogInfo(" --- all OK");
    } catch (const cond::Exception& er) {
      edm::LogError("SiPixelCondObjOfflineBuilder") << er.what() << std::endl;
    } catch (const std::exception& er) {
      edm::LogError("SiPixelCondObjOfflineBuilder") << "caught std::exception " << er.what() << std::endl;
    } catch (...) {
      edm::LogError("SiPixelCondObjOfflineBuilder") << "Funny error" << std::endl;
    }
  }

  // ------------ method called once each job just before starting event loop  ------------
  void SiPixelCondObjOfflineBuilder::beginJob() {
    if (fromFile_) {
      if (loadFromFile()) {
        edm::LogInfo("SiPixelCondObjOfflineBuilder") << " Calibration loaded: Map size " << calmap_.size() << " max "
                                                     << calmap_.max_size() << " " << calmap_.empty() << std::endl;
      }
    }
  }

  // ------------ method called once each job just after ending the event loop  ------------
  void SiPixelCondObjOfflineBuilder::endJob() {}

  bool SiPixelCondObjOfflineBuilder::loadFromFile() {
    float par0, par1;  //,par2,par3;
    int colid, rowid;  //rocid
    std::string name;

    std::ifstream in_file;                          // data file pointer
    in_file.open(fileName_.c_str(), std::ios::in);  // in C++
    if (in_file.bad()) {
      edm::LogError("SiPixelCondObjOfflineBuilder") << "Input file not found" << std::endl;
    }
    if (in_file.eof() != 0) {
      edm::LogError("SiPixelCondObjOfflineBuilder") << in_file.eof() << " " << in_file.gcount() << " " << in_file.fail()
                                                    << " " << in_file.good() << " end of file " << std::endl;
      return false;
    }
    //Load file header
    char line[500];
    for (int i = 0; i < 3; i++) {
      in_file.getline(line, 500, '\n');
      edm::LogInfo("SiPixelCondObjOfflineBuilder") << line << std::endl;
    }
    //Loading calibration constants from file, loop on pixels
    for (int i = 0; i < (52 * 80); i++) {
      in_file >> par0 >> par1 >> name >> colid >> rowid;

      std::cout << " Col " << colid << " Row " << rowid << " P0 " << par0 << " P1 " << par1 << std::endl;

      CalParameters onePix;
      onePix.p0 = par0;
      onePix.p1 = par1;

      // Convert ROC pixel index to channel
      int chan = PixelIndices::pixelToChannelROC(rowid, colid);
      calmap_.insert(std::pair<int, CalParameters>(chan, onePix));
    }

    bool flag;
    if (calmap_.size() == 4160) {
      flag = true;
    } else {
      flag = false;
    }
    return flag;
  }

}  // namespace cms

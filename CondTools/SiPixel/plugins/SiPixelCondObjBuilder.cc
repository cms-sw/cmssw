// -*- C++ -*-
//
// Package:    SiPixelCondObjBuilder
// Class:      SiPixelCondObjBuilder
//
/**\class SiPixelCondObjBuilder SiPixelCondObjBuilder.h SiPixel/test/SiPixelCondObjBuilder.h

 Description: Test analyzer for writing pixel calibration in the DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
// $Id: SiPixelCondObjBuilder.h,v 1.9 2009/05/28 22:12:54 dlange Exp $
//
//

// system includes
#include <memory>
#include <iostream>
#include <string>

// user includes
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationService.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
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

#include "CLHEP/Random/RandGauss.h"

namespace cms {
  class SiPixelCondObjBuilder : public edm::one::EDAnalyzer<> {
  public:
    explicit SiPixelCondObjBuilder(const edm::ParameterSet& iConfig);

    ~SiPixelCondObjBuilder() override = default;
    void beginJob() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    bool loadFromFile();

  private:
    const bool appendMode_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pddToken_;
    std::unique_ptr<SiPixelGainCalibration> SiPixelGainCalibration_;
    SiPixelGainCalibrationService SiPixelGainCalibrationService_;
    const std::string recordName_;

    const double meanPed_;
    const double rmsPed_;
    const double meanGain_;
    const double rmsGain_;
    const double secondRocRowGainOffset_;
    const double secondRocRowPedOffset_;
    const int numberOfModules_;
    const bool fromFile_;
    const std::string fileName_;

    // Internal class
    class CalParameters {
    public:
      float p0;
      float p1;
    };
    // Map for storing calibration constants
    std::map<int, CalParameters, std::less<int> > calmap_;
    PixelIndices* pIndexConverter_;  // Pointer to the index converter
  };
}  // namespace cms

namespace cms {
  SiPixelCondObjBuilder::SiPixelCondObjBuilder(const edm::ParameterSet& iConfig)
      : appendMode_(iConfig.getUntrackedParameter<bool>("appendMode", true)),
        pddToken_(esConsumes()),
        SiPixelGainCalibration_(nullptr),
        SiPixelGainCalibrationService_(iConfig, consumesCollector()),
        recordName_(iConfig.getParameter<std::string>("record")),
        meanPed_(iConfig.getParameter<double>("meanPed")),
        rmsPed_(iConfig.getParameter<double>("rmsPed")),
        meanGain_(iConfig.getParameter<double>("meanGain")),
        rmsGain_(iConfig.getParameter<double>("rmsGain")),
        secondRocRowGainOffset_(iConfig.getParameter<double>("secondRocRowGainOffset")),
        secondRocRowPedOffset_(iConfig.getParameter<double>("secondRocRowPedOffset")),
        numberOfModules_(iConfig.getParameter<int>("numberOfModules")),
        fromFile_(iConfig.getParameter<bool>("fromFile")),
        fileName_(iConfig.getParameter<std::string>("fileName")) {
    ::putenv((char*)"CORAL_AUTH_USER=me");
    ::putenv((char*)"CORAL_AUTH_PASSWORD=test");
  }

  void SiPixelCondObjBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm;
    unsigned int run = iEvent.id().run();
    int nmodules = 0;
    uint32_t nchannels = 0;
    //    int mycol = 415;
    //    int myrow = 159;

    edm::LogInfo("SiPixelCondObjBuilder")
        << "... creating dummy SiPixelGainCalibration Data for Run " << run << "\n " << std::endl;
    //
    // Instantiate Gain calibration offset and define pedestal/gain range
    //
    float mingain = 0;
    float maxgain = 10;
    float minped = 0;
    float maxped = 255;
    SiPixelGainCalibration_ = std::make_unique<SiPixelGainCalibration>(minped, maxped, mingain, maxgain);

    const TrackerGeometry* pDD = &iSetup.getData(pddToken_);
    edm::LogInfo("SiPixelCondObjBuilder") << " There are " << pDD->dets().size() << " detectors" << std::endl;

    for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
      if (dynamic_cast<PixelGeomDetUnit const*>((*it)) != nullptr) {
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
        //edm::LogPrint("SiPixelCondObjBuilder") << " ---> PIXEL DETID " << detid << " Cols " << ncols << " Rows " << nrows << std::endl;

        PixelIndices pIndexConverter(ncols, nrows);

        std::vector<char> theSiPixelGainCalibration;

        // Loop over columns and rows of this DetID
        for (int i = 0; i < ncols; i++) {
          for (int j = 0; j < nrows; j++) {
            nchannels++;

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
              if (rmsPed_ > 0) {
                ped = CLHEP::RandGauss::shoot(meanPed_, rmsPed_);
                while (minped > ped || maxped < ped)
                  ped = CLHEP::RandGauss::shoot(meanPed_, rmsPed_);

              } else
                ped = meanPed_;
              if (rmsGain_ > 0) {
                gain = CLHEP::RandGauss::shoot(meanGain_, rmsGain_);
                while (mingain > gain || maxgain < gain)
                  gain = CLHEP::RandGauss::shoot(meanGain_, rmsGain_);
              } else
                gain = meanGain_;
            }

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

            // 	   if(i==mycol && j==myrow) {
            //	     edm::LogPrint("SiPixelCondObjBuilder") << "       Col "<<i<<" Row "<<j<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
            // 	   }

            // 	   gain =  2.8;
            // 	   ped  = 28.2;

            // Insert data in the container
            SiPixelGainCalibration_->setData(ped, gain, theSiPixelGainCalibration);
          }
        }

        SiPixelGainCalibration::Range range(theSiPixelGainCalibration.begin(), theSiPixelGainCalibration.end());
        if (!SiPixelGainCalibration_->put(detid, range, ncols))
          edm::LogError("SiPixelCondObjBuilder")
              << "[SiPixelCondObjBuilder::analyze] detid already exists" << std::endl;
      }
    }
    edm::LogPrint("SiPixelCondObjBuilder") << " ---> PIXEL Modules  " << nmodules << std::endl;
    edm::LogPrint("SiPixelCondObjBuilder") << " ---> PIXEL Channels " << nchannels << std::endl;

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
    //        //edm::LogPrint("SiPixelCondObjBuilder")<<" PEDESTAL "<< mypedestal<<" GAIN "<<mygain<<std::endl;
    //      }
    //    }
    // Write into DB
    edm::LogInfo(" --- writeing to DB!");
    edm::Service<cond::service::PoolDBOutputService> mydbservice;
    if (!mydbservice.isAvailable()) {
      edm::LogPrint("SiPixelCondObjBuilder") << "Didn't get DB" << std::endl;
      edm::LogError("db service unavailable");
      return;
    } else {
      edm::LogInfo("DB service OK");
    }

    try {
      //     size_t callbackToken = mydbservice->callbackToken("SiPixelGainCalibration");
      //     edm::LogInfo("SiPixelCondObjBuilder")<<"CallbackToken SiPixelGainCalibration "
      //         <<callbackToken<<std::endl;
      //       unsigned long long tillTime;
      //     if ( appendMode_)
      //	 tillTime = mydbservice->currentTime();
      //       else
      //	 tillTime = mydbservice->endOfTime();
      //     edm::LogInfo("SiPixelCondObjBuilder")<<"[SiPixelCondObjBuilder::analyze] tillTime = "
      //         <<tillTime<<std::endl;
      //     mydbservice->newValidityForNewPayload<SiPixelGainCalibration>(
      //           SiPixelGainCalibration_, tillTime , callbackToken);

      if (mydbservice->isNewTagRequest(recordName_)) {
        mydbservice->createOneIOV<SiPixelGainCalibration>(
            *SiPixelGainCalibration_, mydbservice->beginOfTime(), recordName_);
      } else {
        mydbservice->appendOneIOV<SiPixelGainCalibration>(
            *SiPixelGainCalibration_, mydbservice->currentTime(), recordName_);
      }
      edm::LogInfo(" --- all OK");
    } catch (const cond::Exception& er) {
      edm::LogPrint("SiPixelCondObjBuilder") << "Database exception!   " << er.what() << std::endl;
      edm::LogError("SiPixelCondObjBuilder") << er.what() << std::endl;
    } catch (const std::exception& er) {
      edm::LogPrint("SiPixelCondObjBuilder") << "Standard exception!   " << er.what() << std::endl;
      edm::LogError("SiPixelCondObjBuilder") << "caught std::exception " << er.what() << std::endl;
    } catch (...) {
      edm::LogError("SiPixelCondObjBuilder") << "Funny error" << std::endl;
    }
  }

  // ------------ method called once each job just before starting event loop  ------------
  void SiPixelCondObjBuilder::beginJob() {
    if (fromFile_) {
      if (loadFromFile()) {
        edm::LogInfo("SiPixelCondObjBuilder") << " Calibration loaded: Map size " << calmap_.size() << " max "
                                              << calmap_.max_size() << " " << calmap_.empty() << std::endl;
      }
    }
  }

  bool SiPixelCondObjBuilder::loadFromFile() {
    float par0, par1;  //,par2,par3;
    int colid, rowid;  //rocid
    std::string name;

    std::ifstream in_file;                          // data file pointer
    in_file.open(fileName_.c_str(), std::ios::in);  // in C++
    if (in_file.bad()) {
      edm::LogError("SiPixelCondObjBuilder") << "Input file not found" << std::endl;
    }
    if (in_file.eof() != 0) {
      edm::LogError("SiPixelCondObjBuilder") << in_file.eof() << " " << in_file.gcount() << " " << in_file.fail() << " "
                                             << in_file.good() << " end of file " << std::endl;
      return false;
    }
    //Load file header
    char line[500];
    for (int i = 0; i < 3; i++) {
      in_file.getline(line, 500, '\n');
      edm::LogInfo("SiPixelCondObjBuilder") << line << std::endl;
    }
    //Loading calibration constants from file, loop on pixels
    for (int i = 0; i < (52 * 80); i++) {
      in_file >> par0 >> par1 >> name >> colid >> rowid;

      edm::LogPrint("SiPixelCondObjBuilder")
          << " Col " << colid << " Row " << rowid << " P0 " << par0 << " P1 " << par1 << std::endl;

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

using namespace cms;
DEFINE_FWK_MODULE(SiPixelCondObjBuilder);

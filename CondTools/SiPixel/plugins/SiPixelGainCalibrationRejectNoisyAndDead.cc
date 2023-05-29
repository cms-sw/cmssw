// -*- C++ -*-
//
// Package:    InsertNoisyPixelsInDB
// Class:      InsertNoisyPixelsInDB
//
/**\class InsertNoisyPixelsInDB InsertNoisyPixelsInDB.cc CondTools/InsertNoisyPixelsInDB/src/InsertNoisyPixelsInDB.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Romain Rougny
//         Created:  Tue Feb  3 15:18:02 CET 2009
// $Id: SiPixelGainCalibrationRejectNoisyAndDead.cc,v 1.5 2009/10/21 15:53:42 heyburn Exp $
//
//

// system include files
#include <fstream>
#include <map>
#include <memory>
#include <sys/stat.h>
#include <utility>
#include <vector>

// user include files
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// ROOT includes
#include "TH2F.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TKey.h"
#include "TString.h"
#include "TList.h"

class SiPixelGainCalibrationRejectNoisyAndDead : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelGainCalibrationRejectNoisyAndDead(const edm::ParameterSet&);
  ~SiPixelGainCalibrationRejectNoisyAndDead() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pddToken_;
  SiPixelGainCalibrationOfflineService SiPixelGainCalibrationOfflineService_;
  std::unique_ptr<SiPixelGainCalibrationOffline> theGainCalibrationDbInputOffline_;

  SiPixelGainCalibrationForHLTService SiPixelGainCalibrationForHLTService_;
  std::unique_ptr<SiPixelGainCalibrationForHLT> theGainCalibrationDbInputForHLT_;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::map<int, std::vector<std::pair<int, int> > > noisypixelkeeper;
  std::map<int, std::vector<std::pair<int, int> > > insertednoisypixel;
  int nnoisyininput;

  void fillDatabase(const edm::EventSetup&);
  void getNoisyPixels();

  // ----------member data ---------------------------
  const std::string noisypixellist_;
  const int insertnoisypixelsindb_;
  const std::string record_;
  const bool DEBUG;
  float pedlow_;
  float pedhi_;
  float gainlow_;
  float gainhi_;
};

using namespace edm;
using namespace std;

void SiPixelGainCalibrationRejectNoisyAndDead::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("EDAnalyzer to create SiPixelGainCalibration payloads by rejecting noisy and dead ones");
  desc.addUntracked<bool>("debug", false);
  desc.addUntracked<int>("insertNoisyPixelsInDB", 1);
  desc.addUntracked<std::string>("record", "SiPixelGainCalibrationOfflineRcd");
  desc.addUntracked<std::string>("noisyPixelList", "noisypixel.txt");
  descriptions.addWithDefaultLabel(desc);
}

SiPixelGainCalibrationRejectNoisyAndDead::SiPixelGainCalibrationRejectNoisyAndDead(const edm::ParameterSet& iConfig)
    : pddToken_(esConsumes()),
      SiPixelGainCalibrationOfflineService_(iConfig, consumesCollector()),
      SiPixelGainCalibrationForHLTService_(iConfig, consumesCollector()),
      noisypixellist_(iConfig.getUntrackedParameter<std::string>("noisyPixelList", "noisypixel.txt")),
      insertnoisypixelsindb_(iConfig.getUntrackedParameter<int>("insertNoisyPixelsInDB", 1)),
      record_(iConfig.getUntrackedParameter<std::string>("record", "SiPixelGainCalibrationOfflineRcd")),
      DEBUG(iConfig.getUntrackedParameter<bool>("debug", false)) {
  //now do what ever initialization is needed
}

SiPixelGainCalibrationRejectNoisyAndDead::~SiPixelGainCalibrationRejectNoisyAndDead() = default;

// ------------ method called to for each event  ------------
void SiPixelGainCalibrationRejectNoisyAndDead::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (insertnoisypixelsindb_ != 0)
    getNoisyPixels();
  fillDatabase(iSetup);
}

void SiPixelGainCalibrationRejectNoisyAndDead::fillDatabase(const edm::EventSetup& iSetup) {
  if (DEBUG)
    edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
        << "=>=>=>=> Starting the function fillDatabase()" << endl;

  if (record_ != "SiPixelGainCalibrationOfflineRcd" && record_ != "SiPixelGainCalibrationForHLTRcd") {
    edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
        << record_ << " : this record  can't be used !" << std::endl;
    edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
        << "Please select SiPixelGainCalibrationForHLTRcd or SiPixelGainCalibrationOfflineRcd" << std::endl;
    return;
  }

  //Get the Calibration Data
  if (record_ == "SiPixelGainCalibrationOfflineRcd")
    SiPixelGainCalibrationOfflineService_.setESObjects(iSetup);
  if (record_ == "SiPixelGainCalibrationForHLTRcd")
    SiPixelGainCalibrationForHLTService_.setESObjects(iSetup);

  //Get list of ideal detids
  const TrackerGeometry* pDD = &iSetup.getData(pddToken_);
  edm::LogInfo("SiPixelCondObjOfflineBuilder") << " There are " << pDD->dets().size() << " detectors" << std::endl;

  if (record_ == "SiPixelGainCalibrationOfflineRcd") {
    pedlow_ = SiPixelGainCalibrationOfflineService_.getPedLow();
    pedhi_ = SiPixelGainCalibrationOfflineService_.getPedHigh();
    gainlow_ = SiPixelGainCalibrationOfflineService_.getGainLow();
    gainhi_ = SiPixelGainCalibrationOfflineService_.getGainHigh();
  } else if (record_ == "SiPixelGainCalibrationForHLTRcd") {
    pedlow_ = SiPixelGainCalibrationForHLTService_.getPedLow();
    pedhi_ = SiPixelGainCalibrationForHLTService_.getPedHigh();
    gainlow_ = SiPixelGainCalibrationForHLTService_.getGainLow();
    gainhi_ = SiPixelGainCalibrationForHLTService_.getGainHigh();
  }
  if (gainlow_ < 0)
    gainlow_ = 0;
  if (gainhi_ > 21)
    gainhi_ = 21;
  if (pedlow_ < -100)
    pedlow_ = -100;
  if (pedhi_ > 300)
    pedhi_ = 300;

  edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
      << "New payload will have pedlow,hi " << pedlow_ << "," << pedhi_ << " and gainlow,hi " << gainlow_ << ","
      << gainhi_ << endl;
  if (record_ == "SiPixelGainCalibrationOfflineRcd")
    theGainCalibrationDbInputOffline_ =
        std::make_unique<SiPixelGainCalibrationOffline>(pedlow_, pedhi_, gainlow_, gainhi_);
  if (record_ == "SiPixelGainCalibrationForHLTRcd")
    theGainCalibrationDbInputForHLT_ =
        std::make_unique<SiPixelGainCalibrationForHLT>(pedlow_, pedhi_, gainlow_, gainhi_);

  int nnoisy = 0;
  int ndead = 0;

  int detid = 0;

  //checking for noisy pixels that won't be inserted ...
  bool willNoisyPixBeInserted;
  for (std::map<int, std::vector<std::pair<int, int> > >::const_iterator it = noisypixelkeeper.begin();
       it != noisypixelkeeper.end();
       it++) {
    willNoisyPixBeInserted = false;
    for (TrackerGeometry::DetContainer::const_iterator mod = pDD->dets().begin(); mod != pDD->dets().end(); mod++) {
      detid = 0;
      if (dynamic_cast<PixelGeomDetUnit const*>((*mod)) != nullptr)
        detid = ((*mod)->geographicalId()).rawId();
      if (detid == it->first) {
        willNoisyPixBeInserted = true;
        break;
      }
    }
    if (!willNoisyPixBeInserted)
      edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
          << "All Noisy Pixels in detid " << it->first
          << "won't be inserted, check the TrackerGeometry you are using !! You are missing some modules" << endl;
  }

  if (DEBUG)
    edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "=>=>=>=> Starting Loop over all modules" << endl;

  //Looping over all modules
  for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
    detid = 0;
    if (dynamic_cast<PixelGeomDetUnit const*>((*it)) != nullptr)
      detid = ((*it)->geographicalId()).rawId();
    if (detid == 0)
      continue;

    if (DEBUG)
      edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "=>=>=>=> We are in module " << detid << endl;

    // Get the module sizes
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>((*it));
    const PixelTopology& topol = pixDet->specificTopology();
    int nrows = topol.nrows();     // rows in x
    int ncols = topol.ncolumns();  // cols in y

    float ped;
    float gainforthiscol[2];
    float pedforthiscol[2];
    int nusedrows[2];
    int nrowsrocsplit;
    if (record_ == "SiPixelGainCalibrationOfflineRcd")
      nrowsrocsplit = theGainCalibrationDbInputOffline_->getNumberOfRowsToAverageOver();
    if (record_ == "SiPixelGainCalibrationForHLTRcd")
      nrowsrocsplit = theGainCalibrationDbInputForHLT_->getNumberOfRowsToAverageOver();

    std::vector<char> theSiPixelGainCalibrationGainPerColPedPerPixel;
    std::vector<char> theSiPixelGainCalibrationPerCol;

    if (DEBUG)
      edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "=>=>=>=> Starting Loop for each rows/cols " << endl;

    for (int icol = 0; icol <= ncols - 1; icol++) {
      if (DEBUG)
        edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "=>=>=>=> Starting a new column" << endl;

      nusedrows[0] = nusedrows[1] = 0;
      gainforthiscol[0] = gainforthiscol[1] = 0;
      pedforthiscol[0] = pedforthiscol[1] = 0;

      for (int jrow = 0; jrow <= nrows - 1; jrow++) {
        if (DEBUG)
          edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
              << "=>=>=>=> We are in col,row " << icol << "," << jrow << endl;

        ped = 0;

        size_t iglobalrow = 0;
        int noisyPixInRow = 0;
        if (jrow > nrowsrocsplit) {
          iglobalrow = 1;
          noisyPixInRow = 0;
        }

        bool isPixelDeadOld = false;
        bool isColumnDeadOld = false;
        // 	bool isPixelDeadNew = false;
        // 	bool isColumnDeadNew = false;

        bool isPixelNoisyOld = false;
        bool isColumnNoisyOld = false;
        bool isPixelNoisyNew = false;
        bool isColumnNoisyNew = false;

        bool isColumnDead = false;
        bool isColumnNoisy = false;
        bool isPixelDead = false;
        bool isPixelNoisy = false;

        if (DEBUG)
          edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "=>=>=>=> Trying to get gain/ped " << endl;

        try {
          if (record_ == "SiPixelGainCalibrationOfflineRcd") {
            isPixelDeadOld = SiPixelGainCalibrationOfflineService_.isDead(detid, icol, jrow);
            isPixelNoisyOld = SiPixelGainCalibrationOfflineService_.isNoisy(detid, icol, jrow);
            isColumnDeadOld = SiPixelGainCalibrationOfflineService_.isNoisyColumn(detid, icol, jrow);
            isColumnNoisyOld = SiPixelGainCalibrationOfflineService_.isDeadColumn(detid, icol, jrow);
          }
          if (record_ == "SiPixelGainCalibrationForHLTRcd") {
            isColumnDeadOld = SiPixelGainCalibrationForHLTService_.isNoisyColumn(detid, icol, jrow);
            isColumnNoisyOld = SiPixelGainCalibrationForHLTService_.isDeadColumn(detid, icol, jrow);
          }
          if (!isColumnDeadOld && !isColumnNoisyOld) {
            if (record_ == "SiPixelGainCalibrationOfflineRcd")
              gainforthiscol[iglobalrow] = SiPixelGainCalibrationOfflineService_.getGain(detid, icol, jrow);
            if (record_ == "SiPixelGainCalibrationForHLTRcd") {
              gainforthiscol[iglobalrow] = SiPixelGainCalibrationForHLTService_.getGain(detid, icol, jrow);
              pedforthiscol[iglobalrow] = SiPixelGainCalibrationForHLTService_.getPedestal(detid, icol, jrow);
            }
          }
          if (!isPixelDeadOld && !isPixelNoisyOld)
            if (record_ == "SiPixelGainCalibrationOfflineRcd")
              ped = SiPixelGainCalibrationOfflineService_.getPedestal(detid, icol, jrow);
        } catch (const std::exception& er) {
          edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
              << "Problem trying to catch gain/ped from DETID " << detid << " @ col,row " << icol << "," << jrow
              << endl;
        }
        //edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")<<"For DetId "<<detid<<" we found gain : "<<gain<<", pedestal : "<<ped<<std::endl;

        if (DEBUG)
          edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
              << "=>=>=>=> Found gain " << gainforthiscol[iglobalrow] << " and ped " << pedforthiscol[iglobalrow]
              << endl;

        //Check if pixel is in new noisy list
        for (std::map<int, std::vector<std::pair<int, int> > >::const_iterator it = noisypixelkeeper.begin();
             it != noisypixelkeeper.end();
             it++)
          for (unsigned int i = 0; i < (it->second).size(); i++)
            if (it->first == detid && (it->second.at(i)).first == icol && (it->second.at(i)).second == jrow)
              isPixelNoisyNew = true;

        isColumnDead = isColumnDeadOld;
        isPixelDead = isPixelDeadOld;

        if (isPixelNoisyNew)
          noisyPixInRow++;
        if (noisyPixInRow == nrowsrocsplit)
          isColumnNoisyNew = true;

        if (insertnoisypixelsindb_ == 0) {
          isColumnNoisy = isColumnNoisyOld;
          isPixelNoisy = isPixelNoisyOld;
        } else if (insertnoisypixelsindb_ == 1) {
          isPixelNoisy = isPixelNoisyNew;
          isColumnNoisy = isColumnNoisyNew;
        } else if (insertnoisypixelsindb_ == 2) {
          isPixelNoisy = isPixelNoisyNew || isPixelNoisyOld;
          isColumnNoisy = isColumnNoisyNew || isColumnNoisyOld;
        }

        if (isPixelNoisy)
          edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
              << "Inserting a noisy pixel in " << detid << " at col,row " << icol << "," << jrow << endl;
        if (isColumnNoisy)
          edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
              << "Inserting a noisy column in " << detid << " at col,row " << icol << "," << jrow << endl;
        if (isPixelNoisy)
          nnoisy++;
        if (isPixelDead)
          ndead++;

        if (DEBUG)
          edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "=>=>=>=> Now Starting to fill the DB" << endl;

        //**********  Fill the new DB !!

        if (DEBUG)
          edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
              << "=>=>=>=> Filling Pixel Level Calibration" << endl;

        //Set Pedestal
        if (record_ == "SiPixelGainCalibrationOfflineRcd") {
          if (isPixelDead)
            theGainCalibrationDbInputOffline_->setDeadPixel(theSiPixelGainCalibrationGainPerColPedPerPixel);
          else if (isPixelNoisy)
            theGainCalibrationDbInputOffline_->setNoisyPixel(theSiPixelGainCalibrationGainPerColPedPerPixel);
          else
            theGainCalibrationDbInputOffline_->setDataPedestal(ped, theSiPixelGainCalibrationGainPerColPedPerPixel);
        }

        //Set Gain
        if ((jrow + 1) % nrowsrocsplit == 0) {
          if (DEBUG)
            edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
                << "=>=>=>=> Filling Column Level Calibration" << endl;

          if (isColumnDead) {
            if (record_ == "SiPixelGainCalibrationOfflineRcd")
              theGainCalibrationDbInputOffline_->setDeadColumn(nrowsrocsplit,
                                                               theSiPixelGainCalibrationGainPerColPedPerPixel);
            if (record_ == "SiPixelGainCalibrationForHLTRcd")
              theGainCalibrationDbInputForHLT_->setDeadColumn(nrowsrocsplit, theSiPixelGainCalibrationPerCol);
          } else if (isColumnNoisy) {
            if (record_ == "SiPixelGainCalibrationOfflineRcd")
              theGainCalibrationDbInputOffline_->setNoisyColumn(nrowsrocsplit,
                                                                theSiPixelGainCalibrationGainPerColPedPerPixel);
            if (record_ == "SiPixelGainCalibrationForHLTRcd")
              theGainCalibrationDbInputForHLT_->setNoisyColumn(nrowsrocsplit, theSiPixelGainCalibrationPerCol);
          } else {
            if (record_ == "SiPixelGainCalibrationOfflineRcd")
              theGainCalibrationDbInputOffline_->setDataGain(
                  gainforthiscol[iglobalrow], nrowsrocsplit, theSiPixelGainCalibrationGainPerColPedPerPixel);
            if (record_ == "SiPixelGainCalibrationForHLTRcd")
              theGainCalibrationDbInputForHLT_->setData(
                  pedforthiscol[iglobalrow], gainforthiscol[iglobalrow], theSiPixelGainCalibrationPerCol);
          }
        }

        if (DEBUG)
          edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
              << "=>=>=>=> This pixel is finished inserting" << endl;

      }  //end of loop over rows

      if (DEBUG)
        edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
            << "=>=>=>=> This column is finished inserting" << endl;

    }  //end of loop over col

    if (DEBUG)
      edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "=>=>=>=> Loop over rows/cols is finished" << endl;

    if (record_ == "SiPixelGainCalibrationOfflineRcd") {
      SiPixelGainCalibrationOffline::Range offlinerange(theSiPixelGainCalibrationGainPerColPedPerPixel.begin(),
                                                        theSiPixelGainCalibrationGainPerColPedPerPixel.end());
      if (!theGainCalibrationDbInputOffline_->put(detid, offlinerange, ncols))
        edm::LogError("SiPixelGainCalibrationAnalysis")
            << "warning: detid already exists for Offline (gain per col, ped per pixel) calibration database"
            << std::endl;
    }
    if (record_ == "SiPixelGainCalibrationForHLTRcd") {
      SiPixelGainCalibrationOffline::Range hltrange(theSiPixelGainCalibrationPerCol.begin(),
                                                    theSiPixelGainCalibrationPerCol.end());
      if (!theGainCalibrationDbInputForHLT_->put(detid, hltrange, ncols))
        edm::LogError("SiPixelGainCalibrationAnalysis")
            << "warning: detid already exists for HLT (gain per col, ped per col) calibration database" << std::endl;
    }

    if (DEBUG)
      edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "=>=>=>=> This detid is finished inserting" << endl;

  }  //end of loop over Detids

  edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << " --- writing to DB!" << std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogError("db service unavailable");
    return;
  } else {
    if (record_ == "SiPixelGainCalibrationOfflineRcd") {
      edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
          << "now doing SiPixelGainCalibrationOfflineRcd payload..." << std::endl;
      if (mydbservice->isNewTagRequest("SiPixelGainCalibrationOfflineRcd")) {
        mydbservice->createOneIOV<SiPixelGainCalibrationOffline>(
            *theGainCalibrationDbInputOffline_, mydbservice->beginOfTime(), "SiPixelGainCalibrationOfflineRcd");
      } else {
        mydbservice->appendOneIOV<SiPixelGainCalibrationOffline>(
            *theGainCalibrationDbInputOffline_, mydbservice->currentTime(), "SiPixelGainCalibrationOfflineRcd");
      }
    }
    if (record_ == "SiPixelGainCalibrationForHLTRcd") {
      edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
          << "now doing SiPixelGainCalibrationForHLTRcd payload..." << std::endl;
      if (mydbservice->isNewTagRequest("SiPixelGainCalibrationForHLTRcd")) {
        mydbservice->createOneIOV<SiPixelGainCalibrationForHLT>(
            *theGainCalibrationDbInputForHLT_, mydbservice->beginOfTime(), "SiPixelGainCalibrationForHLTRcd");
      } else {
        mydbservice->appendOneIOV<SiPixelGainCalibrationForHLT>(
            *theGainCalibrationDbInputForHLT_, mydbservice->currentTime(), "SiPixelGainCalibrationForHLTRcd");
      }
    }
  }

  edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << " ---> SUMMARY :" << std::endl;
  edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
      << " File had   " << nnoisyininput << " noisy pixels" << std::endl;

  edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << " DB has now " << nnoisy << " noisy pixels" << std::endl;
  edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << " DB has now " << ndead << " dead pixels" << std::endl;
}

void SiPixelGainCalibrationRejectNoisyAndDead::getNoisyPixels() {
  nnoisyininput = 0;

  ifstream in;
  struct stat Stat;
  if (stat(noisypixellist_.c_str(), &Stat)) {
    edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "No file named " << noisypixellist_ << std::endl;
    edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
        << "If you don't want to insert noisy pixel flag, disable it using tag insertNoisyPixelsInDB " << std::endl;
    return;
  }

  in.open(noisypixellist_.c_str());
  if (in.is_open()) {
    TString line;
    edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead") << "opened" << endl;
    char linetmp[201];
    while (in.getline(linetmp, 200)) {
      line = linetmp;
      if (line.Contains("OFFLINE")) {
        line.Remove(0, line.First(",") + 9);
        TString detidstring = line;
        detidstring.Remove(line.First(" "), line.Sizeof());

        line.Remove(0, line.First(",") + 20);
        TString col = line;
        col.Remove(line.First(","), line.Sizeof());
        line.Remove(0, line.First(",") + 1);
        TString row = line;
        row.Remove(line.First(" "), line.Sizeof());

        edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")
            << "Found noisy pixel in DETID " << detidstring << " col,row " << col << "," << row << std::endl;
        nnoisyininput++;

        std::vector<std::pair<int, int> > tempvec;
        if (noisypixelkeeper.find(detidstring.Atoi()) != noisypixelkeeper.end())
          tempvec = (noisypixelkeeper.find(detidstring.Atoi()))->second;

        std::pair<int, int> temppair(col.Atoi(), row.Atoi());
        tempvec.push_back(temppair);
        noisypixelkeeper[detidstring.Atoi()] = tempvec;
      }
    }
  }
  /*
  for(std::map <int,std::vector<std::pair<int,int> > >::const_iterator it=noisypixelkeeper.begin();it!=noisypixelkeeper.end();it++) 
    for(int i=0;i<(it->second).size();i++)
      edm::LogPrint("SiPixelGainCalibrationRejectNoisyAndDead")<<it->first<<"  "<<(it->second.at(i)).first<<"  "<<(it->second.at(i)).second<<std::endl;
  */
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelGainCalibrationRejectNoisyAndDead);

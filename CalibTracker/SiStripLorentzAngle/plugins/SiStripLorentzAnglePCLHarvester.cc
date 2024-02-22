// -*- C++ -*-
//
// Package:    CalibTracker/SiStripLorentzAnglePCLHarvester
// Class:      SiStripLorentzAnglePCLHarvester
//
/**\class SiStripLorentzAnglePCLHarvester SiStripLorentzAnglePCLHarvester.cc CalibTracker/SiStripLorentzAngle/plugins/SiStripLorentzAnglePCLHarvester.cc
 Description: reads the intermediate ALCAPROMPT DQMIO-like dataset and performs the fitting of the SiStrip Lorentz Angle in the Prompt Calibration Loop
*/
//
// Original Author:  mmusich
//         Created:  Sat, 29 May 2021 14:46:19 GMT
//
//

// system includes
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// user includes
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngleCalibrationHelpers.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngleCalibrationStruct.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//------------------------------------------------------------------------------
class SiStripLorentzAnglePCLHarvester : public DQMEDHarvester {
public:
  SiStripLorentzAnglePCLHarvester(const edm::ParameterSet&);
  ~SiStripLorentzAnglePCLHarvester() override = default;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  std::string getStem(const std::string& histoName, bool isFolder);

  // es tokens
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoEsTokenBR_, topoEsTokenER_;
  const edm::ESGetToken<SiStripLatency, SiStripLatencyRcd> latencyToken_;
  const edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleDepRcd> siStripLAEsToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;

  // member data

  bool mismatchedBField_;
  bool mismatchedLatency_;

  const bool debug_;
  SiStripLorentzAngleCalibrationHistograms iHists_;

  const std::string dqmDir_;
  const std::vector<double> fitRange_;
  const std::string recordName_;
  float theMagField_{0.f};

  static constexpr float teslaToInverseGeV_ = 2.99792458e-3f;
  std::pair<double, double> theFitRange_{0., 0.};

  const SiStripLorentzAngle* currentLorentzAngle_;
  std::unique_ptr<TrackerTopology> theTrackerTopology_;
};

//------------------------------------------------------------------------------
SiStripLorentzAnglePCLHarvester::SiStripLorentzAnglePCLHarvester(const edm::ParameterSet& iConfig)
    : geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      topoEsTokenBR_(esConsumes<edm::Transition::BeginRun>()),
      topoEsTokenER_(esConsumes<edm::Transition::EndRun>()),
      latencyToken_(esConsumes<edm::Transition::BeginRun>()),
      siStripLAEsToken_(esConsumes<edm::Transition::BeginRun>()),
      magneticFieldToken_(esConsumes<edm::Transition::BeginRun>()),
      mismatchedBField_{false},
      mismatchedLatency_{false},
      debug_(iConfig.getParameter<bool>("debugMode")),
      dqmDir_(iConfig.getParameter<std::string>("dqmDir")),
      fitRange_(iConfig.getParameter<std::vector<double>>("fitRange")),
      recordName_(iConfig.getParameter<std::string>("record")) {
  // initialize the fit range
  if (fitRange_.size() == 2) {
    theFitRange_.first = fitRange_[0];
    theFitRange_.second = fitRange_[1];
  } else {
    throw cms::Exception("SiStripLorentzAnglePCLHarvester") << "Too many fit range parameters specified";
  }

  // first ensure DB output service is available
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())
    throw cms::Exception("SiStripLorentzAnglePCLHarvester") << "PoolDBService required";
}

//------------------------------------------------------------------------------
void SiStripLorentzAnglePCLHarvester::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // geometry
  const TrackerGeometry* geom = &iSetup.getData(geomEsToken_);
  const TrackerTopology* tTopo = &iSetup.getData(topoEsTokenBR_);

  const MagneticField* magField = &iSetup.getData(magneticFieldToken_);
  currentLorentzAngle_ = &iSetup.getData(siStripLAEsToken_);

  // B-field value
  // inverseBzAtOriginInGeV() returns the inverse of field z component for this map in GeV
  // for the conversion please consult https://github.com/cms-sw/cmssw/blob/master/MagneticField/Engine/src/MagneticField.cc#L17
  // theInverseBzAtOriginInGeV = 1.f / (at0z * 2.99792458e-3f);
  // ==> at0z = 1.f / (theInverseBzAtOriginInGeV * 2.99792458e-3f)

  theMagField_ = 1.f / (magField->inverseBzAtOriginInGeV() * teslaToInverseGeV_);

  if (iHists_.bfield_.empty()) {
    iHists_.bfield_ = siStripLACalibration::fieldAsString(theMagField_);
  } else {
    if (iHists_.bfield_ != siStripLACalibration::fieldAsString(theMagField_)) {
      mismatchedBField_ = true;
    }
  }

  const SiStripLatency* apvlat = &iSetup.getData(latencyToken_);
  if (iHists_.apvmode_.empty()) {
    iHists_.apvmode_ = siStripLACalibration::apvModeAsString(apvlat);
  } else {
    if (iHists_.apvmode_ != siStripLACalibration::apvModeAsString(apvlat)) {
      mismatchedLatency_ = true;
    }
  }

  auto dets = geom->detsTIB();
  dets.insert(dets.end(), geom->detsTID().begin(), geom->detsTID().end());
  dets.insert(dets.end(), geom->detsTOB().begin(), geom->detsTOB().end());
  dets.insert(dets.end(), geom->detsTEC().begin(), geom->detsTEC().end());

  for (auto det : dets) {
    auto detid = det->geographicalId().rawId();
    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(geom->idToDet(det->geographicalId()));
    if (stripDet) {
      iHists_.la_db_[detid] = currentLorentzAngle_->getLorentzAngle(detid);
      iHists_.moduleLocationType_[detid] = siStripLACalibration::moduleLocationType(detid, tTopo);
    }
  }
}

//------------------------------------------------------------------------------
void SiStripLorentzAnglePCLHarvester::endRun(edm::Run const& run, edm::EventSetup const& isetup) {
  if (!theTrackerTopology_) {
    theTrackerTopology_ = std::make_unique<TrackerTopology>(isetup.getData(topoEsTokenER_));
  }
}

//------------------------------------------------------------------------------
void SiStripLorentzAnglePCLHarvester::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  if (mismatchedBField_) {
    throw cms::Exception("SiStripLorentzAnglePCLHarvester") << "Trying to harvest runs with different B-field values!";
  }

  if (mismatchedLatency_) {
    throw cms::Exception("SiStripLorentzAnglePCLHarvester") << "Trying to harvest runs with diffent APV modes!";
  }

  // go in the right directory
  iGetter.cd();
  std::string bvalue = (iHists_.bfield_ == "3.8") ? "B-ON" : "B-OFF";
  std::string folderToHarvest = fmt::format("{}/{}_{}", dqmDir_, bvalue, iHists_.apvmode_);
  edm::LogPrint(moduleDescription().moduleName()) << "Harvesting in " << folderToHarvest;
  iGetter.setCurrentFolder(folderToHarvest);

  // fill in the module types
  iHists_.nlayers_["TIB"] = 4;
  iHists_.nlayers_["TOB"] = 6;
  iHists_.modtypes_.push_back("s");
  iHists_.modtypes_.push_back("a");

  std::vector<std::string> MEtoHarvest = {"tanthcosphtrk_nstrip",
                                          "thetatrk_nstrip",
                                          "tanthcosphtrk_var2",
                                          "tanthcosphtrk_var3",
                                          "thcosphtrk_var2",
                                          "thcosphtrk_var3"};

  // prepare the histograms to be harvested
  for (auto& layers : iHists_.nlayers_) {
    std::string subdet = layers.first;
    for (int l = 1; l <= layers.second; ++l) {
      for (auto& t : iHists_.modtypes_) {
        // do not fill stereo where there aren't
        if (l > 2 && t == "s")
          continue;
        std::string locationtype = Form("%s_L%d%s", subdet.c_str(), l, t.c_str());
        for (const auto& toHarvest : MEtoHarvest) {
          const char* address = Form(
              "%s/%s/L%d/%s_%s", folderToHarvest.c_str(), subdet.c_str(), l, locationtype.c_str(), toHarvest.c_str());

          LogDebug(moduleDescription().moduleName()) << "harvesting at: " << address << std::endl;

          iHists_.h2_[Form("%s_%s", locationtype.c_str(), toHarvest.c_str())] = iGetter.get(address);
          if (iHists_.h2_[Form("%s_%s", locationtype.c_str(), toHarvest.c_str())] == nullptr) {
            edm::LogError(moduleDescription().moduleName())
                << "could not retrieve: " << Form("%s_%s", locationtype.c_str(), toHarvest.c_str());
          }
        }
      }
    }
  }

  // book the summary output histograms
  iBooker.setCurrentFolder(fmt::format("{}Harvesting/LorentzAngleMaps", dqmDir_));

  // Define a lambda function to extract the second element and add it to the accumulator
  auto sumValues = [](int accumulator, const std::pair<std::string, int>& element) {
    return accumulator + element.second;
  };

  // Use std::accumulate to sum the values
  int totalLayers = std::accumulate(iHists_.nlayers_.begin(), iHists_.nlayers_.end(), 0, sumValues);

  // Lambda expression to set bin labels for a TH2F histogram
  auto setHistoLabels = [](TH2F* histogram, const std::map<std::string, int>& nlayers) {
    // Set common options
    histogram->SetOption("colz1");  // don't fill empty bins
    histogram->SetStats(false);
    histogram->GetYaxis()->SetLabelSize(0.05);
    histogram->GetXaxis()->SetLabelSize(0.05);

    // Set bin labels for the X-axis
    histogram->GetXaxis()->SetBinLabel(1, "r-#phi");
    histogram->GetXaxis()->SetBinLabel(2, "stereo");

    // Set bin labels for the Y-axis
    int binCounter = 1;
    for (const auto& subdet : {"TIB", "TOB"}) {
      for (int layer = 1; layer <= nlayers.at(subdet); ++layer) {
        std::string label = Form("%s L%d", subdet, layer);
        histogram->GetYaxis()->SetBinLabel(binCounter++, label.c_str());
      }
    }
    histogram->GetXaxis()->LabelsOption("h");
  };

  std::string d_name = "h2_byLayerSiStripLA";
  std::string d_text = "SiStrip tan#theta_{LA}/B;module type (r-#phi/stereo);layer number;tan#theta_{LA}/B [1/T]";
  iHists_.h2_byLayerLA_ =
      iBooker.book2D(d_name.c_str(), d_text.c_str(), 2, -0.5, 1.5, totalLayers, -0.5, totalLayers - 0.5);

  setHistoLabels(iHists_.h2_byLayerLA_->getTH2F(), iHists_.nlayers_);

  d_name = "h2_byLayerSiStripLADiff";
  d_text = "SiStrip #Delta#mu_{H}/#mu_{H};module type (r-#phi/stereo);ladder number;#Delta#mu_{H}/#mu_{H} [%%]";
  iHists_.h2_byLayerDiff_ =
      iBooker.book2D(d_name.c_str(), d_text.c_str(), 2, -0.5, 1.5, totalLayers, -0.5, totalLayers - 0.5);

  setHistoLabels(iHists_.h2_byLayerDiff_->getTH2F(), iHists_.nlayers_);

  // prepare the profiles
  for (const auto& ME : iHists_.h2_) {
    if (!ME.second)
      continue;
    // draw colored 2D plots
    ME.second->setOption("colz");
    TProfile* hp = (TProfile*)ME.second->getTH2F()->ProfileX();
    iBooker.setCurrentFolder(folderToHarvest + "/" + getStem(ME.first, /* isFolder = */ true));
    iHists_.p_[hp->GetName()] = iBooker.bookProfile(hp->GetName(), hp);
    iHists_.p_[hp->GetName()]->setAxisTitle(ME.second->getAxisTitle(2), 2);
    delete hp;
  }

  if (iHists_.p_.empty()) {
    edm::LogError(moduleDescription().moduleName()) << "None of the input histograms could be retrieved. Aborting";
    return;
  }

  std::map<std::string, std::pair<double, double>> LAMap_;

  // do the fits
  for (const auto& prof : iHists_.p_) {
    //fit only this type of profile
    if ((prof.first).find("thetatrk_nstrip") != std::string::npos) {
      // Create the TF1 function

      // fitting range (take full axis by default)
      double low = prof.second->getAxisMin(1);
      double high = prof.second->getAxisMax(1);
      if (theFitRange_.first > theFitRange_.second) {
        low = theFitRange_.first;
        high = theFitRange_.second;
      }

      TF1* fitFunc = new TF1("fitFunc", siStripLACalibration::fitFunction, low, high, 3);

      // Fit the function to the data
      prof.second->getTProfile()->Fit(fitFunc, "F");  // "F" option performs a least-squares fit

      // Get the fit results
      Double_t a_fit = fitFunc->GetParameter(0);
      Double_t thetaL_fit = fitFunc->GetParameter(1);
      Double_t b_fit = fitFunc->GetParameter(2);

      Double_t a_fitError = fitFunc->GetParError(0);
      Double_t thetaL_fitError = fitFunc->GetParError(1);
      Double_t b_fitError = fitFunc->GetParError(2);

      if (debug_) {
        edm::LogInfo(moduleDescription().moduleName())
            << prof.first << " fit result: "
            << " a=" << a_fit << " +/ " << a_fitError << " theta_L=" << thetaL_fit << " +/- " << thetaL_fitError
            << " b=" << b_fit << " +/- " << b_fitError;
      }

      LAMap_[getStem(prof.first, /* isFolder = */ false)] = std::make_pair(thetaL_fit, thetaL_fitError);
    }
  }

  if (debug_) {
    for (const auto& element : LAMap_) {
      edm::LogInfo(moduleDescription().moduleName())
          << element.first << " thetaLA = " << element.second.first << "+/-" << element.second.second;
    }
  }

  // now prepare the output LA
  std::shared_ptr<SiStripLorentzAngle> OutLorentzAngle = std::make_shared<SiStripLorentzAngle>();

  bool isPayloadChanged{false};
  std::vector<std::pair<int, int>> treatedIndices;
  for (const auto& loc : iHists_.moduleLocationType_) {
    if (debug_) {
      edm::LogInfo(moduleDescription().moduleName()) << "modId: " << loc.first << " " << loc.second;
    }

    if (!(loc.second).empty() && theMagField_ != 0.f) {
      OutLorentzAngle->putLorentzAngle(loc.first, std::abs(LAMap_[loc.second].first / theMagField_));
    } else {
      OutLorentzAngle->putLorentzAngle(loc.first, iHists_.la_db_[loc.first]);
    }

    // if the location is  not assigned (e.g. TID or TEC) continue
    if ((loc.second).empty()) {
      continue;
    }

    const auto& index2D = siStripLACalibration::locationTypeIndex(loc.second);
    LogDebug("SiStripLorentzAnglePCLHarvester")
        << loc.first << " : " << loc.second << " index: " << index2D.first << "-" << index2D.second << std::endl;

    // check if the location exists, otherwise throw!
    if (index2D != std::make_pair(-1, -1)) {
      // Check if index2D is in treatedIndices
      // Do not fill the control plots more than necessary (i.e. 1 entry per "partition")
      auto it = std::find(treatedIndices.begin(), treatedIndices.end(), index2D);
      if (it == treatedIndices.end()) {
        // control plots
        LogTrace("SiStripLorentzAnglePCLHarvester") << "accepted " << loc.first << " : " << loc.second << " bin ("
                                                    << index2D.first << "," << index2D.second << ")";

        const auto& outputLA = OutLorentzAngle->getLorentzAngle(loc.first);
        const auto& inputLA = currentLorentzAngle_->getLorentzAngle(loc.first);

        LogTrace("SiStripLorentzAnglePCLHarvester") << "inputLA: " << inputLA << " outputLA: " << outputLA;

        iHists_.h2_byLayerLA_->setBinContent(index2D.first, index2D.second, outputLA);

        float deltaMuHoverMuH = (inputLA != 0.f) ? (inputLA - outputLA) / inputLA : 0.f;
        iHists_.h2_byLayerDiff_->setBinContent(index2D.first, index2D.second, deltaMuHoverMuH * 100.f);
        treatedIndices.emplace_back(index2D);

        // Check if the delta is different from zero
        // if none of the locations has a non-zero diff
        // will not write out the payload.
        if (deltaMuHoverMuH != 0.f) {
          isPayloadChanged = true;
          LogDebug("SiStripLorentzAnglePCLHarvester")
              << "accepted " << loc.first << " : " << loc.second << " bin (" << index2D.first << "," << index2D.second
              << ") " << deltaMuHoverMuH;
        }

      }  // if the index has not been treated already
    } else {
      throw cms::Exception("SiStripLorentzAnglePCLHarvester")
          << "Trying to fill an inexistent module location from " << loc.second << "!";
    }  //
  }    // ends loop on location types

  if (isPayloadChanged) {
    // fill the DB object record
    edm::Service<cond::service::PoolDBOutputService> mydbservice;
    if (mydbservice.isAvailable()) {
      try {
        mydbservice->writeOneIOV(*OutLorentzAngle, mydbservice->currentTime(), recordName_);
      } catch (const cond::Exception& er) {
        edm::LogError("SiStripLorentzAngleDB") << er.what();
      } catch (const std::exception& er) {
        edm::LogError("SiStripLorentzAngleDB") << "caught std::exception " << er.what();
      }
    } else {
      edm::LogError("SiStripLorentzAngleDB") << "Service is unavailable!";
    }
  } else {
    edm::LogPrint("SiStripLorentzAngleDB")
        << "****** WARNING ******\n* " << __PRETTY_FUNCTION__
        << "\n* There is no new valid measurement to append!\n* Will NOT update the DB!\n*********************";
  }
}

//------------------------------------------------------------------------------
std::string SiStripLorentzAnglePCLHarvester::getStem(const std::string& histoName, bool isFolder) {
  std::vector<std::string> tokens;

  std::string output{};
  // Create a string stream from the input string
  std::istringstream iss(histoName);

  std::string token;
  while (std::getline(iss, token, '_')) {
    // Add each token to the vector
    tokens.push_back(token);
  }

  if (isFolder) {
    output = tokens[0] + "/" + tokens[1];
    output.pop_back();
  } else {
    output = tokens[0] + "_" + tokens[1];
  }
  return output;
}

//------------------------------------------------------------------------------
void SiStripLorentzAnglePCLHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Harvester module of the SiStrip Lorentz Angle PCL monitoring workflow");
  desc.add<bool>("debugMode", false)->setComment("determines if it's in debug mode");
  desc.add<std::string>("dqmDir", "AlCaReco/SiStripLorentzAngle")->setComment("the directory of PCL Worker output");
  desc.add<std::vector<double>>("fitRange", {0., 0.})->setComment("range of depths to perform the LA fit");
  desc.add<std::string>("record", "SiStripLorentzAngleRcd")->setComment("target DB record");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(SiStripLorentzAnglePCLHarvester);

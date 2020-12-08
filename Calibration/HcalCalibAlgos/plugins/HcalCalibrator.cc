// -*- C++ -*-F
//---------------------------------------------------------------------------------
// Package:    HcalCalibrator
// Class:      HcalCalibrator
//
/**\class HcalCalibrator HcalCalibrator.cc MyPackages/HcalCalibrator/src/HcalCalibrator.cc

Description: <one line class summary>

Implementation:

This is an interface to run the the hcal calibration code for isolated tracsk and dijets.
It takes tha parameters for the calibration settings from a python file and passes them 
to the actual calibration code in "endJob()". 



*/
//
// Original Author:  "Anton Anastassov"
//         Created:  Tue Sept 24 09:13:48 CDT 2008
//
//
//_________________________________________________________________________________

// system include files
#include <memory>
#include <fstream>
#include <iostream>

// user include files

#include "Calibration/HcalCalibAlgos/plugins/HcalCalibrator.h"
#include "Calibration/HcalCalibAlgos/interface/hcalCalib.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//--------------
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"

#include "TFile.h"

// constructor

HcalCalibrator::HcalCalibrator(const edm::ParameterSet& conf)
    : mInputFileList(conf.getUntrackedParameter<std::string>("inputFileList")),
      //  mOutputFile(conf.getUntrackedParameter<std::string>("outputFile")),
      mCalibType(conf.getUntrackedParameter<std::string>("calibType")),
      mCalibMethod(conf.getUntrackedParameter<std::string>("calibMethod")),
      mMinTargetE(conf.getUntrackedParameter<double>("minTargetE")),
      mMaxTargetE(conf.getUntrackedParameter<double>("maxTargetE")),
      mMinCellE(conf.getUntrackedParameter<double>("minCellE")),
      mMinEOverP(conf.getUntrackedParameter<double>("minEOverP")),
      mMaxEOverP(conf.getUntrackedParameter<double>("maxEOverP")),
      mMaxTrkEmE(conf.getUntrackedParameter<double>("maxTrkEmE")),
      mMaxEtThirdJet(conf.getUntrackedParameter<double>("maxEtThirdJet")),
      mMinDPhiDiJets(conf.getUntrackedParameter<double>("minDPhiDiJets")),
      mSumDepths(conf.getUntrackedParameter<bool>("sumDepths")),
      mSumSmallDepths(conf.getUntrackedParameter<bool>("sumSmallDepths")),
      mCombinePhi(conf.getUntrackedParameter<bool>("combinePhi")),
      mHbClusterSize(conf.getUntrackedParameter<int>("hbClusterSize")),
      mHeClusterSize(conf.getUntrackedParameter<int>("heClusterSize")),

      mUseConeClustering(conf.getUntrackedParameter<bool>("useConeClustering")),
      mMaxConeDist(conf.getUntrackedParameter<double>("maxConeDist")),

      mCalibAbsIEtaMax(conf.getUntrackedParameter<int>("calibAbsIEtaMax")),
      mCalibAbsIEtaMin(conf.getUntrackedParameter<int>("calibAbsIEtaMin")),
      mMaxProbeJetEmFrac(conf.getUntrackedParameter<double>("maxProbeJetEmFrac")),
      mMaxTagJetEmFrac(conf.getUntrackedParameter<double>("maxTagJetEmFrac")),
      mMaxTagJetAbsEta(conf.getUntrackedParameter<double>("maxTagJetAbsEta")),
      mMinTagJetEt(conf.getUntrackedParameter<double>("minTagJetEt")),
      mMinProbeJetAbsEta(conf.getUntrackedParameter<double>("minProbeJetAbsEta")),
      mPhiSymCorFileName(conf.getUntrackedParameter<std::string>("phiSymCorFileName")),
      mApplyPhiSymCorFlag(conf.getUntrackedParameter<bool>("applyPhiSymCorFlag")),
      mOutputCorCoefFileName(conf.getUntrackedParameter<std::string>("outputCorCoefFileName")),
      mHistoFileName(conf.getUntrackedParameter<std::string>("histoFileName")) {
  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
}

// destructor

HcalCalibrator::~HcalCalibrator() {}

// ------------ method called to for each event  ------------

void HcalCalibrator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  mTheCaloGeometry = &iSetup.getData(tok_geom_);
  mTheHcalTopology = &iSetup.getData(tok_htopo_);
}

// ------------ method called once each job just before starting event loop  ------------

void HcalCalibrator::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------

void HcalCalibrator::endJob() {
  if (mCalibType != "DI_JET" && mCalibType != "ISO_TRACK") {
    std::cout << "\n\nUnknown calibration type " << mCalibType << std::endl;
    std::cout << "Please select ISO_TRACK or DI_JET in the python file." << std::endl;
    return;
  }

  if (mCalibMethod != "L3" && mCalibMethod != "MATRIX_INV_OF_ETA_AVE" && mCalibMethod != "L3_AND_MTRX_INV") {
    std::cout << "\n\nUnknown calibration method " << mCalibMethod << std::endl;
    std::cout << "Supported methods for IsoTrack calibration are: L3, MATRIX_INV_OF_ETA_AVE, L3_AND_MTRX_INV"
              << std::endl;
    std::cout << "For DiJets the supported method is L3" << std::endl;
    return;
  }

  if (mCalibType == "DI_JET" && mCalibMethod != "L3") {
    std::cout << "\n\nDiJet calibration can use only the L3 method. Please change the python file." << std::endl;
    return;
  }

  if (mCalibAbsIEtaMin < 1 || mCalibAbsIEtaMax > 41 || mCalibAbsIEtaMin > mCalibAbsIEtaMax) {
    std::cout
        << "\n\nInvalid ABS(iEta) calibration range. Check calibAbsIEtaMin and calibAbsIEtaMax in the python file."
        << std::endl;
    return;
  }

  hcalCalib* calibrator = new hcalCalib();

  // set the parameters controlling the calibratoration

  calibrator->SetCalibType(mCalibType);
  calibrator->SetCalibMethod(mCalibMethod);
  calibrator->SetMinTargetE(mMinTargetE);
  calibrator->SetMaxTargetE(mMaxTargetE);
  calibrator->SetMaxEtThirdJet(mMaxEtThirdJet);
  calibrator->SetMinDPhiDiJets(mMinDPhiDiJets);
  calibrator->SetSumDepthsFlag(mSumDepths);
  calibrator->SetSumSmallDepthsFlag(mSumSmallDepths);
  calibrator->SetCombinePhiFlag(mCombinePhi);
  calibrator->SetMinCellE(mMinCellE);
  calibrator->SetMinEOverP(mMinEOverP);
  calibrator->SetMaxEOverP(mMaxEOverP);
  calibrator->SetMaxTrkEmE(mMaxTrkEmE);
  calibrator->SetHbClusterSize(mHbClusterSize);
  calibrator->SetHeClusterSize(mHeClusterSize);

  calibrator->SetUseConeClustering(mUseConeClustering);
  calibrator->SetConeMaxDist(mMaxConeDist);

  calibrator->SetCalibAbsIEtaMax(mCalibAbsIEtaMax);
  calibrator->SetCalibAbsIEtaMin(mCalibAbsIEtaMin);
  calibrator->SetMaxProbeJetEmFrac(mMaxProbeJetEmFrac);
  calibrator->SetMaxTagJetEmFrac(mMaxTagJetEmFrac);
  calibrator->SetMaxTagJetAbsEta(mMaxTagJetAbsEta);

  calibrator->SetMinTagJetEt(mMinTagJetEt);

  calibrator->SetMinProbeJetAbsEta(mMinProbeJetAbsEta);
  calibrator->SetApplyPhiSymCorFlag(mApplyPhiSymCorFlag);
  calibrator->SetPhiSymCorFileName(mPhiSymCorFileName);
  calibrator->SetOutputCorCoefFileName(mOutputCorCoefFileName);

  calibrator->SetHistoFileName(mHistoFileName);

  calibrator->SetCaloGeometry(mTheCaloGeometry, mTheHcalTopology);

  std::ifstream inputFileList;  // contains list of input root files

  TString files = mInputFileList;
  inputFileList.open(files.Data());

  std::vector<TString> inputFiles;
  while (!inputFileList.eof()) {
    TString fileName;
    inputFileList >> fileName;
    if (!fileName.BeginsWith("#") && !fileName.Contains(" ") && fileName != "")
      inputFiles.push_back(fileName);
  }
  inputFileList.close();

  std::cout << "\nInput files for processing:" << std::endl;
  for (std::vector<TString>::iterator it = inputFiles.begin(); it != inputFiles.end(); ++it) {
    std::cout << "file: " << it->Data() << std::endl;
  }
  std::cout << std::endl;

  TChain* fChain = new TChain("hcalCalibTree");

  for (std::vector<TString>::iterator f_it = inputFiles.begin(); f_it != inputFiles.end(); ++f_it) {
    fChain->Add(f_it->Data());
  }

  fChain->Process(calibrator);

  if (fChain)
    delete fChain;
  delete calibrator;

  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalCalibrator);

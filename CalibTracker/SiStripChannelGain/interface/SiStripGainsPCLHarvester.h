// -*- C++ -*-
//
// Package:    CalibTracker/SiStripChannelGain
// Class:      SiStripGainsPCLHarvester
//
/**\class SiStripGainsPCLHarvester SiStripGainsPCLHarvester.cc 
 Description: Harvests output of SiStripGainsPCLWorker and creates histograms and Gains Payload
 
*/
//
//  Original Author: L. Quertermont (calibration algorithm)
//  Contributors:    M. Verzetti    (data access)
//                   A. Di Mattia   (PCL multi stream processing and monitoring)
//                   M. Delcourt    (monitoring)
//                   M. Musich      (migration to thread-safe DQMStore access)
//
//  Created:  Wed, 12 Apr 2017 14:46:48 GMT
//

// CMSSW includes
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

// user includes
#include "CalibTracker/SiStripChannelGain/interface/APVGainStruct.h"

// ROOT includes
#include "TH1F.h"
#include "TH2S.h"
#include "TProfile.h"
#include "TF1.h"

// System includes
#include <unordered_map>

class SiStripGainsPCLHarvester : public DQMEDHarvester {
public:
  explicit SiStripGainsPCLHarvester(const edm::ParameterSet& ps);
  void beginRun(edm::Run const& run, edm::EventSetup const& isetup) override;
  void endRun(edm::Run const& run, edm::EventSetup const& isetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void checkBookAPVColls(const edm::EventSetup& setup);
  void dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) override;
  void gainQualityMonitor(DQMStore::IBooker& ibooker_, const MonitorElement* Charge_Vs_Index) const;
  void storeGainsTree(const TAxis* chVsIdxXaxis) const;
  int statCollectionFromMode(const char* tag) const;

  void algoComputeMPVandGain(const MonitorElement* Charge_Vs_Index);
  void getPeakOfLandau(TH1* InputHisto,
                       double* FitResults,
                       double LowRange = 50,
                       double HighRange = 5400,
                       bool gaussianConvolution = false);
  bool IsGoodLandauFit(double* FitResults);

  bool produceTagFilter(const MonitorElement* Charge_Vs_Index);
  std::unique_ptr<SiStripApvGain> getNewObject(const MonitorElement* Charge_Vs_Index);

  bool doStoreOnDB;
  bool doChargeMonitorPerPlane; /*!< Charge monitor per detector plane */
  bool storeGainsTree_;
  unsigned int GOOD;
  unsigned int BAD;
  unsigned int MASKED;

  double tagCondition_NClusters;
  double tagCondition_GoodFrac;

  int NStripAPVs;
  int NPixelDets;
  double MinNrEntries;

  std::string m_Record;

  std::string m_DQMdir;                  /*!< DQM folder hosting the charge statistics and the monitor plots */
  std::string m_calibrationMode;         /*!< Type of statistics for the calibration */
  std::vector<std::string> VChargeHisto; /*!< Charge monitor plots to be output */

  std::vector<std::string> dqm_tag_;

  int CalibrationLevel;

  const TrackerGeometry* bareTkGeomPtr_ = nullptr;  // ugly hack to fill APV colls only once, but checks
  std::unique_ptr<TrackerTopology> tTopo_;

  std::vector<std::shared_ptr<stAPVGain> > APVsCollOrdered;
  std::unordered_map<unsigned int, std::shared_ptr<stAPVGain> > APVsColl;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoTokenBR_, tTopoTokenER_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  edm::ESGetToken<SiStripGain, SiStripGainRcd> gainToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;

  // fit options
  bool fit_gaussianConvolution_ = false;
  bool fit_gaussianConvolutionTOBL56_ = false;
  bool fit_dataDrivenRange_ = false;
};

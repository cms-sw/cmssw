// -*- C++ -*-
//
// Package:    SiPixelOfflineCalibAnalysisBase
// Class:      SiPixelOfflineCalibAnalysisBase
// 
/**\class SiPixelOfflineCalibAnalysisBase SiPixelOfflineCalibAnalysisBase.h CalibTracker/SiPixelTools/src/SiPixelOfflineCalibAnalysisBase.h

 Description: Base class for Pixel calibrations

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Evan Klose Friis
//    additions by:  Freya Blekman
//         Created:  Tue Nov  6 17:27:19 CET 2007
// $Id: SiPixelOfflineCalibAnalysisBase.h,v 1.19 2011/06/29 10:29:35 eulisse Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigifwd.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"

#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

#include "TF1.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

#include <map>
#include <vector>
#include <iostream>
#include <string>
//
// class decleration
//


class SiPixelOfflineCalibAnalysisBase : public edm::EDAnalyzer {
public:
  explicit SiPixelOfflineCalibAnalysisBase(const edm::ParameterSet&); 
  ~SiPixelOfflineCalibAnalysisBase();
  
  //no argument constructor only used to throw exception in the case of derived 
  //class constructor not calling SiPixelOfflineCalibAnalysisBase(const edm::ParameterSet&)
  SiPixelOfflineCalibAnalysisBase();
  
  //pure virtual function, called after each DetID loaded
  virtual bool doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix);
  
  
  //translate DetID to human readable string
  std::string translateDetIdToString(uint32_t detid);
  
  //booking DQM histograms (for dynamic histogram creation)
  
  MonitorElement* 			bookDQMHistogram1D(uint32_t detid, std::string name, std::string title, int nchX, double lowX, double highX);  
  MonitorElement*			bookDQMHistogram1D(uint32_t detid, std::string name, std::string title, int nchX, float *xbinsize);  //variable size bins
  MonitorElement* 			bookDQMHistogram2D(uint32_t detid, std::string name, std::string title, int nchX, double lowX, double highX, int nchY, double lowY, double highY);
  
  MonitorElement*                       bookDQMHistoPlaquetteSummary2D(uint32_t detid, std::string name,std::string title); // take the detid to determine the size of rows and columns, this saves looking up everything in the cabling map by the user. 
  void                                  addTF1ToDQMMonitoringElement(MonitorElement *ele, TF1 *func);
  
  bool				        setDQMDirectory(std::string dirName);	
  bool				        setDQMDirectory(uint32_t detID); //automatically create directory hierachy based on DetID
  static TF1*                           fitFunction_;
  static const std::vector<short>*      getVcalValues();
  std::vector<uint32_t> &               getRunNumbers() { return runnumbers_;}
  
protected:

  //calibration parameters from calib.dat/DB
  edm::ESHandle<SiPixelCalibConfiguration> calib_;
  edm::ESHandle<TrackerGeometry> geom_;
  edm::ESHandle<SiPixelFedCablingMap> theCablingMap_;
  
  std::string calibrationMode_;
  short nTriggers_;
  static std::vector<short> vCalValues_;
  uint32_t & EventNumber() { return eventCounter_;}
  

private:
  
  SiPixelFolderOrganizer* folderMaker_;
  DQMStore* daqBE_;
  SiPixelHistogramId * theHistogramIdWorker_;
  std::string outputFileName_;
  bool createOutputFile_;

  std::vector<uint32_t> runnumbers_;
  uint32_t eventCounter_;

  //store set of detIDs that have been encountered
  //second int argument can be a derived class result flag
  std::map<uint32_t, int> detIdsEntered_;
  std::map<uint32_t, std::string> detIdNames_;
  
  edm::InputTag siPixelCalibDigiProducer_;
  
  //the beginJob is used to load the calib database.  It then calls the pure
  //virtual calibrationSetup() function.  Derived classes should put beginJob functionality there.
  virtual void beginRun(const edm::Run &, const edm::EventSetup &);
  void beginRun(const edm::EventSetup& iSetup);
  void beginJob();
  
  //calibrationSetup will be used by derived classes
  virtual void calibrationSetup(const edm::EventSetup& iSetup);
  
  // pure virtual function, checks if the calibration analyzer is of the right type wrt the calibration information in the database (as defined in the calibrationMode_ variable). Should be implemented in each analyzer, if false will not do anything in analyze loop. default returns true.
  virtual bool checkCorrectCalibrationType();
  
  //called when new DetID discovered
  virtual void newDetID(uint32_t detid);
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  // the endJob method is used to save things like histograms etc.
  // for additional actions derived classes should use the calibrationEnd() method for endJob functionality.
  void endJob() ;
  // calibrationEnd() will be used by derived classes
  virtual void calibrationEnd() ;
 
  // checkPixel returns whether a particular pixel is to be expected during the entire run..
  bool checkPixel(uint32_t detid, short row, short column);
};


// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      HistogramBase
//
/**\class HistogramBase DQM/SiStripMonitorHardware/interface/HistogramBase.hh

 Description: DQM source application to produce data integrety histograms for SiStrip data
*/
//
// Original Author:  Nicholas Cripps in plugin file
//         Created:  2008/09/16
// Modified by    :  Anne-Marie Magnan, code copied from plugin to this class
//

#ifndef DQM_SiStripMonitorHardware_HistogramBase_HH
#define DQM_SiStripMonitorHardware_HistogramBase_HH

#include <sstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class HistogramBase {
public:
  struct HistogramConfig {
    HistogramConfig() : globalswitchon(true) {}
    MonitorElement* monitorEle;
    bool enabled;
    unsigned int nBins;
    double min;
    double max;
    bool globalswitchon;
  };

  virtual ~HistogramBase(){};

  //initialise histograms
  //make it pure abstract: implementation in derived class.
  virtual void initialise(const edm::ParameterSet& iConfig, std::ostringstream* pDebugStream) = 0;

  //fill a histogram if the pointer is not NULL (ie if it has been booked)
  static void fillHistogram(HistogramConfig& histogram, double value, double weight = 1.);

  //fill a histogram if the pointer is not NULL (ie if it has been booked)
  static void fillHistogram(MonitorElement* histogram, double value, double weight = 1.);
  //fill a 2D profile trend histogram if the pointer is not NULL (ie if it has been booked)
  static void fillHistogram2D(HistogramConfig& histogram, double value, double trendVar, double weight = 1.);
  //fill tkHistoMap of percentage of bad channels per module
  static void fillTkHistoMap(TkHistoMap* aMap, uint32_t& detid, float value);

  //if one of more histo map, return name and pointer
  virtual bool tkHistoMapEnabled(unsigned int aIndex = 0) = 0;

  virtual TkHistoMap* tkHistoMapPointer(unsigned int aIndex = 0) = 0;

  //load the config for a histogram from PSet called <configName>HistogramConfig (writes a debug message to stream if pointer is non-NULL)
  void getConfigForHistogram(HistogramConfig& aConfig,
                             const std::string& configName,
                             const edm::ParameterSet& psetContainingConfigPSet,
                             std::ostringstream* pDebugStream);

  //book an individual hiostogram if enabled in config
  void bookHistogram(DQMStore::IBooker&,
                     HistogramConfig& aConfig,
                     const std::string& name,
                     const std::string& title,
                     const unsigned int nBins,
                     const double min,
                     const double max,
                     const std::string& xAxisTitle);

  //book an individual hiostogram if enabled in config
  void bookHistogram(DQMStore::IBooker&,
                     HistogramConfig& aConfig,
                     MonitorElement*& aHist,
                     const std::string& name,
                     const std::string& title,
                     const unsigned int nBins,
                     const double min,
                     const double max,
                     const std::string& xAxisTitle);

  //book an individual hiostogram if enabled in config
  void book2DHistogram(DQMStore::IBooker&,
                       HistogramConfig& aConfig,
                       const std::string& name,
                       const std::string& title,
                       const unsigned int nBins,
                       const double min,
                       const double max,
                       const unsigned int nBinsY,
                       const double minY,
                       const double maxY,
                       const std::string& xAxisTitle,
                       const std::string& yAxisTitle);

  //book an individual hiostogram if enabled in config
  void book2DHistogram(DQMStore::IBooker&,
                       HistogramConfig& aConfig,
                       MonitorElement*& aHist,
                       const std::string& name,
                       const std::string& title,
                       const unsigned int nBins,
                       const double min,
                       const double max,
                       const unsigned int nBinsY,
                       const double minY,
                       const double maxY,
                       const std::string& xAxisTitle,
                       const std::string& yAxisTitle);

  //same but using binning from config
  void bookHistogram(DQMStore::IBooker&,
                     HistogramConfig& aConfig,
                     const std::string& name,
                     const std::string& title,
                     const std::string& xAxisTitle);

  void bookProfile(DQMStore::IBooker&,
                   HistogramConfig& aConfig,
                   const std::string& name,
                   const std::string& title,
                   const unsigned int nBins,
                   const double min,
                   const double max,
                   const double minY,
                   const double maxY,
                   const std::string& xAxisTitle,
                   const std::string& yAxisTitle);

  void bookProfile(DQMStore::IBooker&,
                   HistogramConfig& aConfig,
                   const std::string& name,
                   const std::string& title,
                   const double minY,
                   const double maxY,
                   const std::string& xAxisTitle,
                   const std::string& yAxisTitle);
  void bookProfile2D(DQMStore::IBooker&,
                     HistogramConfig& aConfig,
                     const std::string& name,
                     const std::string& title,
                     const unsigned int nBinsx,
                     const double xmin,
                     const double xmax,
                     const unsigned int nBinsy,
                     const double ymin,
                     const double ymax,
                     const std::string& xAxisTitle,
                     const std::string& yAxisTitle);

  void bookProfile2D(DQMStore::IBooker&,
                     HistogramConfig& aConfig,
                     const std::string& name,
                     const std::string& title,
                     const unsigned int nBinsy,
                     const double ymin,
                     const double ymax,
                     const std::string& xAxisTitle,
                     const std::string& yAxisTitle);

protected:
private:
};  //class

#endif  //DQM_SiStripMonitorHardware_HistogramBase_HH

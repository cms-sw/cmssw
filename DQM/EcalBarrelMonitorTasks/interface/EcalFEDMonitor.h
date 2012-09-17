#ifndef EcalFEDMonitor_H
#define EcalFEDMonitor_H

/*
 * \file EcalFEDMonitor.h
 *
 * $Date: 2012/09/17 16:46:00 $
 * $Revision: 1.1 $
 * \author Y. Iiyama
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"

class MonitorElement;

class EcalFEDMonitor : public edm::EDAnalyzer{
 public:
  EcalFEDMonitor(const edm::ParameterSet&);
  ~EcalFEDMonitor() {}

 private:
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginRun(const edm::Run&, const edm::EventSetup&);

  void initialize();

  enum MEs {
    kEBOccupancy,
    kEBFatal,
    kEBNonFatal,
    kEEOccupancy,
    kEEFatal,
    kEENonFatal,
    nMEs
  };

  bool initialized_;
  std::string folderName_;

  edm::InputTag FEDRawDataTag_;
  edm::InputTag gainErrorsTag_;
  edm::InputTag chIdErrorsTag_;
  edm::InputTag gainSwitchErrorsTag_;
  edm::InputTag towerIdErrorsTag_;
  edm::InputTag blockSizeErrorsTag_;

  std::vector<MonitorElement*> MEs_;
};

#endif

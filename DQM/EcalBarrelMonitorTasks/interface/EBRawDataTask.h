#ifndef EBRawDataTask_H
#define EBRawDataTask_H

/*
 * \file EBRawDataTask.h
 *
 * $Date: 2012/05/14 20:36:37 $
 * $Revision: 1.13 $
 * \author E. Di Marco
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBRawDataTask : public edm::EDAnalyzer {

public:

/// Constructor
EBRawDataTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBRawDataTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(void);

/// EndJob
void endJob(void);

/// BeginLuminosityBlock
void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup);

/// EndLuminosityBlock
void endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup);

/// BeginRun
void beginRun(const edm::Run & r, const edm::EventSetup & c);

/// EndRun
void endRun(const edm::Run & r, const edm::EventSetup & c);

/// Reset
void reset(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

 std::string subfolder_;

bool enableCleanup_;

bool mergeRuns_;

edm::InputTag FEDRawDataCollection_;
edm::InputTag EcalRawDataCollection_;

MonitorElement* meEBCRCErrors_;
MonitorElement* meEBEventTypePreCalibrationBX_;
MonitorElement* meEBEventTypeCalibrationBX_;
MonitorElement* meEBEventTypePostCalibrationBX_;
MonitorElement* meEBRunNumberErrors_;
MonitorElement* meEBOrbitNumberErrors_;
MonitorElement* meEBTriggerTypeErrors_;
MonitorElement* meEBCalibrationEventErrors_;
MonitorElement* meEBL1ADCCErrors_;
MonitorElement* meEBBunchCrossingDCCErrors_;
MonitorElement* meEBL1AFEErrors_;
MonitorElement* meEBBunchCrossingFEErrors_;
MonitorElement* meEBL1ATCCErrors_;
MonitorElement* meEBBunchCrossingTCCErrors_;
MonitorElement* meEBL1ASRPErrors_;
MonitorElement* meEBBunchCrossingSRPErrors_;

MonitorElement* meEBSynchronizationErrorsByLumi_;

 MonitorElement* meEBSynchronizationErrorsTrend_;

 int ls_;

 float fatalErrors_;

bool init_;

float calibrationBX_; 

enum activeEVM { TCS, FDLEVM };

};

#endif

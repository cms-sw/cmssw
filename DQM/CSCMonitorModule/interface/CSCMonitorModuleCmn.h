/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModuleCmn.h
 *
 *    Description:  Updated CSC Monitor module
 *
 *        Version:  1.0
 *        Created:  11/13/2008 01:36:45 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */


#ifndef CSCMonitorModuleCmn_H
#define CSCMonitorModuleCmn_H

/// Global stuff
#include <iostream>

/// DQM Framework stuff
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

/// CSC Framework stuff
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

/// CSCDQM Framework stuff
#include "DQM/CSCMonitorModule/interface/CSCDQM_HistoType.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Collection.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"

/// Local stuff
#include "DQM/CSCMonitorModule/interface/CSCMonitorObject.h"

/// Local Constants
static const char PARAM_BOOKING_FILE[]   = "BookingFile";

class CSCMonitorModuleCmn: public edm::EDAnalyzer {
 
  public:

    CSCMonitorModuleCmn(const edm::ParameterSet& ps);
    virtual ~CSCMonitorModuleCmn();

    const bool getEMUHisto(const cscdqm::EMUHistoType& histo, CSCMonitorObject* mo);
    const bool getDDUHisto(const cscdqm::DDUHistoType& histo, CSCMonitorObject* mo);
    const bool getCSCHisto(const cscdqm::CSCHistoType& histo, CSCMonitorObject* mo);
    const bool getEffParamHisto(const std::string& paramName, CSCMonitorObject* mo);

    void getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition);
    const uint32_t getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const;
    const bool nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const;

  protected:

    void beginJob(const edm::EventSetup& c);
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    void setup();
    void analyze(const edm::Event& e, const edm::EventSetup& c) ;
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;
    void endRun(const edm::Run& r, const edm::EventSetup& c);
    void endJob();

  public:

    cscdqm::Collection collection;

};

#endif

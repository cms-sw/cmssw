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
#include <cstring>
//#include <boost/shared_ptr.hpp>

/// DQM Framework stuff
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

/// CSC Framework stuff
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

/// CSCDQM Framework stuff
#include "DQM/CSCMonitorModule/interface/CSCDQM_EventProcessor.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Collection.h"

/// Local stuff
#include "DQM/CSCMonitorModule/interface/CSCMonitorObject.h"

/// Local Constants
static const char PARAM_BOOKING_FILE[]   = "BookingFile";
static const char DIR_ROOT[]             = "CSC/";
static const char DIR_SUMMARY[]          = "CSC/Summary/";
static const char DIR_EVENTINFO[]        = "CSC/EventInfo/";

class CSCMonitorModuleCmn: public edm::EDAnalyzer, public cscdqm::HistoProvider {
 
  public:

    CSCMonitorModuleCmn(const edm::ParameterSet& ps);
    virtual ~CSCMonitorModuleCmn();

    const bool getEMUHisto(const cscdqm::EMUHistoType& histo, cscdqm::MonitorObject* mo);
    const bool getDDUHisto(const cscdqm::DDUHistoType& histo, cscdqm::MonitorObject* mo);
    const bool getCSCHisto(const cscdqm::CSCHistoType& histo, cscdqm::MonitorObject* mo);
    const bool getEffParamHisto(const std::string& paramName, cscdqm::MonitorObject* mo);

    void getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition);
    const uint32_t getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const;
    const bool nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const;

    cscdqm::MonitorObject* bookInt (const std::string &name) {
      return new CSCMonitorObject(*dbe->bookInt(name));
    }
    cscdqm::MonitorObject* bookFloat (const std::string &name) {
      return new CSCMonitorObject(*dbe->bookFloat(name));
    }
    cscdqm::MonitorObject* bookString (const std::string &name, const std::string &value) {
      return new CSCMonitorObject(*dbe->bookString(name, value));
    }
    cscdqm::MonitorObject* book1D (const std::string &name, const std::string &title, int nchX, double lowX, double highX) {
      return new CSCMonitorObject(*dbe->book1D(name, title, nchX, lowX, highX));
    }
    cscdqm::MonitorObject* book2D (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY) {
      return new CSCMonitorObject(*dbe->book2D(name, title, nchX, lowX, highX, nchY, lowY, highY));
    }
    cscdqm::MonitorObject* book3D (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, int nchZ, double lowZ, double highZ) {
      return new CSCMonitorObject(*dbe->book3D(name, title, nchX, lowX, highX, nchY, lowY, highY, nchZ, lowZ, highZ));
    }
    cscdqm::MonitorObject* bookProfile (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, const char *option = "s") {
      return new CSCMonitorObject(*dbe->bookProfile(name, title, nchX, lowX, highX, nchY, lowY, highY, option));
    }
    cscdqm::MonitorObject* bookProfile2D (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, int nchZ, double lowZ, double highZ, const char *option = "s") {
      return new CSCMonitorObject(*dbe->bookProfile2D(name, title, nchX, lowX, highX, nchY, lowY, highY, nchZ, lowZ, highZ, option));
    }

  protected:

    void beginJob(const edm::EventSetup& c);
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    void setup();
    void analyze(const edm::Event& e, const edm::EventSetup& c) ;
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;
    void endRun(const edm::Run& r, const edm::EventSetup& c);
    void endJob();

  private:

    cscdqm::Collection     *collection;
    cscdqm::EventProcessor *processor;
    DQMStore               *dbe;

};

#endif

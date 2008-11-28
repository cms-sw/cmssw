/*
 * =====================================================================================
 *
 *       Filename:  EventProcessor.cc
 *
 *    Description:  EventProcessor Object
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:47:11 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCDQM_EventProcessor.h"

namespace cscdqm {


  EventProcessor::EventProcessor(MonitorObjectProvider* p_provider) {

    provider = p_provider;

    nEvents = 0;
    nBadEvents = 0;
    nGoodEvents = 0;
    nCSCEvents = 0;
    unpackedDMBcount = 0;

    fFirstEvent = true;
    fCloseL1As = true;

    dduCheckMask = 0xFFFFFFFF;
    binCheckMask = 0xFFFFFFFF;
    dduBinCheckMask = 0x02080016;

  }


  EventProcessor::~EventProcessor() {
  }


  void EventProcessor::blockHisto(const HistoName& histo) {
    blocked.insert(histo);
  }


  const bool EventProcessor::histoBlocked(const HistoName& histo) const {
    std::set<HistoName>::iterator found = blocked.find(histo);
    return (found != blocked.end());
  }


  const bool EventProcessor::getEMUHisto(const HistoName& histo, MonitorObject*& me) {
    if (histoBlocked(histo)) return false;
    EMUHistoType histoT(histo);
    return provider->getHisto(histoT, me);
  }


  const bool EventProcessor::getDDUHisto(const int dduID, const HistoName& histo, MonitorObject*& me) {
    if (histoBlocked(histo)) return false;
    DDUHistoType histoT(histo, dduID);
    return provider->getHisto(histoT, me);
  }


  const bool EventProcessor::getCSCHisto(const int crateID, const int dmbSlot, const HistoName& histo, MonitorObject*& me) {
    if (histoBlocked(histo)) return false;
    CSCHistoType histoT(histo, crateID, dmbSlot);
    return provider->getHisto(histoT, me);
  }


  const bool EventProcessor::getCSCHisto(const int crateID, const int dmbSlot, const HistoName& histo, MonitorObject*& me, const int adId) {
    if (histoBlocked(histo)) return false;
    CSCHistoType histoT(histo, crateID, dmbSlot, adId);
    return provider->getHisto(histoT, me);
  }


  const bool EventProcessor::getParHisto(const std::string& name, MonitorObject*& me) {
    const HistoName histo = const_cast<char*>(name.c_str());
    if (histoBlocked(histo)) return false;
    ParHistoType histoT(histo);
    return provider->getHisto(histoT, me);
  }


  void EventProcessor::setBinCheckerCRC(const BinCheckerCRCType crc, const bool value) {
    switch (crc) {
      case ALCT:
        binChecker.crcALCT(value);
      case CFEB:
        binChecker.crcCFEB(value);
      case TMB:
        binChecker.crcTMB(value);
    };
  }


  void EventProcessor::setBinCheckerOutput(const bool value) {
    if (value) {
      binChecker.output1().show();
      binChecker.output2().show();
    } else {
      binChecker.output1().hide();
      binChecker.output2().hide();
    }
  }

}

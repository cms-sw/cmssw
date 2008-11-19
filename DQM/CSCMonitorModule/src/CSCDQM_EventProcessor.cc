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


  EventProcessor::EventProcessor(HistoProvider* p_histoProvider) {

    histoProvider = p_histoProvider;

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


  void EventProcessor::blockHisto(const HistoType histo) {
    blocked.insert(histo);
  }


  const bool EventProcessor::histoNotBlocked(const HistoType histo) const {
    std::set<HistoType>::iterator found = blocked.find(histo);
    return (found != blocked.end());
  }


  const bool EventProcessor::getEMUHisto(const HistoType histo, MonitorObject* me, const bool ref) {
    if (!ref && !histoNotBlocked(histo)) return false;
    EMUHistoType histoT;
    histoT.histoId = histo;
    histoT.reference = ref;
    histoT.tag = TAG_EMU;
    return histoProvider->getEMUHisto(histoT, me);
  }


  const bool EventProcessor::getDDUHisto(const int dduID, const HistoType histo, MonitorObject* me, const bool ref) {
    if (!ref && !histoNotBlocked(histo)) return false;
    DDUHistoType histoT;
    histoT.histoId = histo;
    histoT.dduId = dduID;
    histoT.reference = ref;
    histoT.tag = Form(TAG_DDU, dduID);
    return histoProvider->getDDUHisto(histoT, me);
  }


  const bool EventProcessor::getCSCHisto(const int crateID, const int dmbSlot, const HistoType histo, MonitorObject* me, const int adId, const bool ref) {
    if (!ref && !histoNotBlocked(histo)) return false;
    CSCHistoType histoT;
    histoT.histoId = histo;
    histoT.crateId = crateID;
    histoT.dmbId = dmbSlot;
    histoT.addId = adId;
    histoT.reference = ref;
    histoT.tag = Form(TAG_CSC, crateID, dmbSlot);
    return histoProvider->getCSCHisto(histoT, me);
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

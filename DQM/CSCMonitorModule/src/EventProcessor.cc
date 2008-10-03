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

#include "DQM/CSCMonitorModule/interface/EventProcessor.h"

namespace cscdqm {

  template <class METype, class HPType>
  EventProcessor<METype, HPType>::EventProcessor(HPType*& p_histoProvider) {
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

  template <class METype, class HPType>
  EventProcessor<METype, HPType>::~EventProcessor() {
  }

  template <class METype, class HPType>
  void EventProcessor<METype, HPType>::blockHisto(const HistoType histo) {
    blocked.insert(histo);
  }

  template <class METype, class HPType>
  const bool EventProcessor<METype, HPType>::histoNotBlocked(const HistoType histo) const {
    std::set<HistoType>::iterator found = blocked.find(histo);
    return (found != blocked.end());
  }

  template <class METype, class HPType>
  const bool EventProcessor<METype, HPType>::getEMUHisto(const HistoType histo, METype* me) {
    if (!histoNotBlocked(histo)) return false;
    return histoProvider->getEMUHisto(histo, me);
  }

  template <class METype, class HPType>
  const bool EventProcessor<METype, HPType>::getDDUHisto(const int dduID, const HistoType histo, METype* me) {
    if (!histoNotBlocked(histo)) return false;
    return histoProvider->getDDUHisto(dduID, histo, me);
  }

  template <class METype, class HPType>
  const bool EventProcessor<METype, HPType>::getCSCHisto(const int crateID, const int dmbSlot, const HistoType histo, METype* me) {
    if (!histoNotBlocked(histo)) return false;
    return histoProvider->getCSCHisto(crateID, dmbSlot, histo, me);
  }

  template <class METype, class HPType>
  void EventProcessor<METype, HPType>::setBinCheckerCRC(const BinCheckerCRCType crc, const bool value) {
    switch (crc) {
      case ALCT:
        binChecker.crcALCT(value);
      case CFEB:
        binChecker.crcCFEB(value);
      case TMB:
        binChecker.crcTMB(value);
    };
  }

  template <class METype, class HPType>
  void EventProcessor<METype, HPType>::setBinCheckerOutput(const bool value) {
    if (value) {
      binChecker.output1().show();
      binChecker.output2().show();
    } else {
      binChecker.output1().hide();
      binChecker.output2().hide();
    }
  }

}

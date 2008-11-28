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


  EventProcessor::EventProcessor(const Configuration* p_config) {

    config = p_config;
    provider = config->provider;

    nEvents = 0;
    nBadEvents = 0;
    nGoodEvents = 0;
    nCSCEvents = 0;
    unpackedDMBcount = 0;

    fFirstEvent = true;
    fCloseL1As = true;

  }

  const bool EventProcessor::getEMUHisto(const HistoName& histo, MonitorObject*& me) {
    EMUHistoType histoT(histo);
    return provider->getHisto(histoT, me);
  }


  const bool EventProcessor::getDDUHisto(const int dduID, const HistoName& histo, MonitorObject*& me) {
    DDUHistoType histoT(histo, dduID);
    return provider->getHisto(histoT, me);
  }


  const bool EventProcessor::getCSCHisto(const int crateID, const int dmbSlot, const HistoName& histo, MonitorObject*& me) {
    CSCHistoType histoT(histo, crateID, dmbSlot);
    return provider->getHisto(histoT, me);
  }


  const bool EventProcessor::getCSCHisto(const int crateID, const int dmbSlot, const HistoName& histo, MonitorObject*& me, const int adId) {
    CSCHistoType histoT(histo, crateID, dmbSlot, adId);
    return provider->getHisto(histoT, me);
  }


  const bool EventProcessor::getParHisto(const std::string& name, MonitorObject*& me) {
    const HistoName histo = const_cast<char*>(name.c_str());
    ParHistoType histoT(histo);
    return provider->getHisto(histoT, me);
  }

}

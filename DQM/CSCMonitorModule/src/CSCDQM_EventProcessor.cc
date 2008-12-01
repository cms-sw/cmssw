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


  EventProcessor::EventProcessor(Configuration* const p_config) {

    config = p_config;
    provider = config->provider;

    fFirstEvent = true;
    fCloseL1As = true;

  }

  const bool EventProcessor::getEMUHisto(const HistoName& histo, MonitorObject*& me) {
    EMUHistoType histoT(histo);
    return config->getHisto(histoT, me);
  }


  const bool EventProcessor::getDDUHisto(const HistoName& histo, const int dduID, MonitorObject*& me) {
    DDUHistoType histoT(histo, dduID);
    return config->getHisto(histoT, me);
  }


  const bool EventProcessor::getCSCHisto(const HistoName& histo, const int crateID, const int dmbSlot, MonitorObject*& me) {
    CSCHistoType histoT(histo, crateID, dmbSlot);
    return config->getHisto(histoT, me);
  }


  const bool EventProcessor::getCSCHisto(const HistoName& histo, const int crateID, const int dmbSlot, const int adId, MonitorObject*& me) {
    CSCHistoType histoT(histo, crateID, dmbSlot, adId);
    return config->getHisto(histoT, me);
  }


  const bool EventProcessor::getParHisto(const std::string& name, MonitorObject*& me) {
    const HistoName histo = const_cast<char*>(name.c_str());
    ParHistoType histoT(histo);
    return config->getHisto(histoT, me);
  }

}

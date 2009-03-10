/*
 * =====================================================================================
 *
 *       Filename:  CSCDcsInfo.cc
 *
 *    Description:  CSC Daq Information Implementaion
 *
 *        Version:  1.0
 *        Created:  12/09/2008 10:55:59 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCDcsInfo.h"

using namespace std;
using namespace edm;

CSCDcsInfo::CSCDcsInfo(const edm::ParameterSet& ps) {
   
}

void CSCDcsInfo::beginJob(const edm::EventSetup& iSetup){

  dbe = Service<DQMStore>().operator->();
       
  dbe->setCurrentFolder("CSC/EventInfo/DCSContents");
  MonitorElement* dcs = dbe->bookFloat("CSCDCS");
  dcs->Fill(1.0);
}


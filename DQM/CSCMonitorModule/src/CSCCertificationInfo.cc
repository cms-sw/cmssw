/*
 * =====================================================================================
 *
 *       Filename:  CSCCertificationInfo.cc
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

#include "DQM/CSCMonitorModule/interface/CSCCertificationInfo.h"

using namespace std;
using namespace edm;

CSCCertificationInfo::CSCCertificationInfo(const edm::ParameterSet& ps) {
   
}

void CSCCertificationInfo::beginJob(const edm::EventSetup& iSetup){

  dbe = Service<DQMStore>().operator->();
       
  dbe->setCurrentFolder("CSC/EventInfo/CertificationContents");
  MonitorElement* dcs = dbe->bookFloat("CSCDCS");
  dcs->Fill(1.0);
}


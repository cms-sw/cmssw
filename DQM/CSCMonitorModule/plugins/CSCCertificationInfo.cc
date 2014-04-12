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

#include "CSCCertificationInfo.h"

using namespace std;
using namespace edm;

CSCCertificationInfo::CSCCertificationInfo(const edm::ParameterSet& ps) {
   
}

void CSCCertificationInfo::beginJob(){

  dbe = Service<DQMStore>().operator->();

  dbe->setCurrentFolder("CSC/EventInfo/CertificationContents");
  mos.insert(std::make_pair("CSC_SideMinus", dbe->bookFloat("CSC_SideMinus")));
  mos.insert(std::make_pair("CSC_SideMinus_Station01", dbe->bookFloat("CSC_SideMinus_Station01")));
  mos.insert(std::make_pair("CSC_SideMinus_Station01_Ring01", dbe->bookFloat("CSC_SideMinus_Station01_Ring01")));
  mos.insert(std::make_pair("CSC_SideMinus_Station01_Ring02", dbe->bookFloat("CSC_SideMinus_Station01_Ring02")));
  mos.insert(std::make_pair("CSC_SideMinus_Station01_Ring03", dbe->bookFloat("CSC_SideMinus_Station01_Ring03")));
  mos.insert(std::make_pair("CSC_SideMinus_Station02", dbe->bookFloat("CSC_SideMinus_Station02")));
  mos.insert(std::make_pair("CSC_SideMinus_Station02_Ring01", dbe->bookFloat("CSC_SideMinus_Station02_Ring01")));
  mos.insert(std::make_pair("CSC_SideMinus_Station02_Ring02", dbe->bookFloat("CSC_SideMinus_Station02_Ring02")));
  mos.insert(std::make_pair("CSC_SideMinus_Station03", dbe->bookFloat("CSC_SideMinus_Station03")));
  mos.insert(std::make_pair("CSC_SideMinus_Station03_Ring01", dbe->bookFloat("CSC_SideMinus_Station03_Ring01")));
  mos.insert(std::make_pair("CSC_SideMinus_Station03_Ring02", dbe->bookFloat("CSC_SideMinus_Station03_Ring02")));
  mos.insert(std::make_pair("CSC_SideMinus_Station04", dbe->bookFloat("CSC_SideMinus_Station04")));
  mos.insert(std::make_pair("CSC_SidePlus", dbe->bookFloat("CSC_SidePlus")));
  mos.insert(std::make_pair("CSC_SidePlus_Station01", dbe->bookFloat("CSC_SidePlus_Station01")));
  mos.insert(std::make_pair("CSC_SidePlus_Station01_Ring01", dbe->bookFloat("CSC_SidePlus_Station01_Ring01")));
  mos.insert(std::make_pair("CSC_SidePlus_Station01_Ring02", dbe->bookFloat("CSC_SidePlus_Station01_Ring02")));
  mos.insert(std::make_pair("CSC_SidePlus_Station01_Ring03", dbe->bookFloat("CSC_SidePlus_Station01_Ring03")));
  mos.insert(std::make_pair("CSC_SidePlus_Station02", dbe->bookFloat("CSC_SidePlus_Station02")));
  mos.insert(std::make_pair("CSC_SidePlus_Station02_Ring01", dbe->bookFloat("CSC_SidePlus_Station02_Ring01")));
  mos.insert(std::make_pair("CSC_SidePlus_Station02_Ring02", dbe->bookFloat("CSC_SidePlus_Station02_Ring02")));
  mos.insert(std::make_pair("CSC_SidePlus_Station03", dbe->bookFloat("CSC_SidePlus_Station03")));
  mos.insert(std::make_pair("CSC_SidePlus_Station03_Ring01", dbe->bookFloat("CSC_SidePlus_Station03_Ring01")));
  mos.insert(std::make_pair("CSC_SidePlus_Station03_Ring02", dbe->bookFloat("CSC_SidePlus_Station03_Ring02")));
  mos.insert(std::make_pair("CSC_SidePlus_Station04", dbe->bookFloat("CSC_SidePlus_Station04")));

  dbe->setCurrentFolder("CSC/EventInfo");
  mos.insert(std::make_pair("CertificationSummary", dbe->bookFloat("CertificationSummary")));

  for (std::map<std::string, MonitorElement*>::iterator it = mos.begin(); it != mos.end(); it++) { 
    it->second->Fill(-1);
  }

}


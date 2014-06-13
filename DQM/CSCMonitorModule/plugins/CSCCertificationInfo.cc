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

/*** No longer triggered for DQMEDHarvester ***/
/*
void CSCCertificationInfo::beginJob(){
 
 for (std::map<std::string, MonitorElement*>::iterator it = mos.begin(); it != mos.end(); it++) { 
    it->second->Fill(-1);
  }

}
*/

// void CSCCertificationInfo::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const &, edm::EventSetup const &)
void CSCCertificationInfo::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter)
{

  

  ibooker.cd();
  ibooker.setCurrentFolder("CSC/EventInfo/CertificationContents");
  
  mos.insert(std::make_pair("CSC_SideMinus", ibooker.bookFloat("CSC_SideMinus")));
  mos.insert(std::make_pair("CSC_SideMinus_Station01", ibooker.bookFloat("CSC_SideMinus_Station01")));
  mos.insert(std::make_pair("CSC_SideMinus_Station01_Ring01", ibooker.bookFloat("CSC_SideMinus_Station01_Ring01")));
  mos.insert(std::make_pair("CSC_SideMinus_Station01_Ring02", ibooker.bookFloat("CSC_SideMinus_Station01_Ring02")));
  mos.insert(std::make_pair("CSC_SideMinus_Station01_Ring03", ibooker.bookFloat("CSC_SideMinus_Station01_Ring03")));
  mos.insert(std::make_pair("CSC_SideMinus_Station02", ibooker.bookFloat("CSC_SideMinus_Station02")));
  mos.insert(std::make_pair("CSC_SideMinus_Station02_Ring01", ibooker.bookFloat("CSC_SideMinus_Station02_Ring01")));
  mos.insert(std::make_pair("CSC_SideMinus_Station02_Ring02", ibooker.bookFloat("CSC_SideMinus_Station02_Ring02")));
  mos.insert(std::make_pair("CSC_SideMinus_Station03", ibooker.bookFloat("CSC_SideMinus_Station03")));
  mos.insert(std::make_pair("CSC_SideMinus_Station03_Ring01", ibooker.bookFloat("CSC_SideMinus_Station03_Ring01")));
  mos.insert(std::make_pair("CSC_SideMinus_Station03_Ring02", ibooker.bookFloat("CSC_SideMinus_Station03_Ring02")));
  mos.insert(std::make_pair("CSC_SideMinus_Station04", ibooker.bookFloat("CSC_SideMinus_Station04")));
  mos.insert(std::make_pair("CSC_SidePlus", ibooker.bookFloat("CSC_SidePlus")));
  mos.insert(std::make_pair("CSC_SidePlus_Station01", ibooker.bookFloat("CSC_SidePlus_Station01")));
  mos.insert(std::make_pair("CSC_SidePlus_Station01_Ring01", ibooker.bookFloat("CSC_SidePlus_Station01_Ring01")));
  mos.insert(std::make_pair("CSC_SidePlus_Station01_Ring02", ibooker.bookFloat("CSC_SidePlus_Station01_Ring02")));
  mos.insert(std::make_pair("CSC_SidePlus_Station01_Ring03", ibooker.bookFloat("CSC_SidePlus_Station01_Ring03")));
  mos.insert(std::make_pair("CSC_SidePlus_Station02", ibooker.bookFloat("CSC_SidePlus_Station02")));
  mos.insert(std::make_pair("CSC_SidePlus_Station02_Ring01", ibooker.bookFloat("CSC_SidePlus_Station02_Ring01")));
  mos.insert(std::make_pair("CSC_SidePlus_Station02_Ring02", ibooker.bookFloat("CSC_SidePlus_Station02_Ring02")));
  mos.insert(std::make_pair("CSC_SidePlus_Station03", ibooker.bookFloat("CSC_SidePlus_Station03")));
  mos.insert(std::make_pair("CSC_SidePlus_Station03_Ring01", ibooker.bookFloat("CSC_SidePlus_Station03_Ring01")));
  mos.insert(std::make_pair("CSC_SidePlus_Station03_Ring02", ibooker.bookFloat("CSC_SidePlus_Station03_Ring02")));
  mos.insert(std::make_pair("CSC_SidePlus_Station04", ibooker.bookFloat("CSC_SidePlus_Station04")));


  ibooker.setCurrentFolder("CSC/EventInfo");
  mos.insert(std::make_pair("CertificationSummary", ibooker.bookFloat("CertificationSummary")));

  for (std::map<std::string, MonitorElement*>::iterator it = mos.begin(); it != mos.end(); it++) { 
    it->second->Fill(-1);
  }

}


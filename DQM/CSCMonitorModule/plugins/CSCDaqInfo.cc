/*
 * =====================================================================================
 *
 *       Filename:  CSCDaqInfo.cc
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

#include "CSCDaqInfo.h"

using namespace std;
using namespace edm;

CSCDaqInfo::CSCDaqInfo(const edm::ParameterSet& ps) {
   
}

void CSCDaqInfo::beginJob() {
/*
  for (std::map<std::string, MonitorElement*>::iterator it = mos.begin(); it != mos.end(); it++) { 
    it->second->Fill(-1);
  }
*/
};

void CSCDaqInfo::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const &, edm::EventSetup const &)
{

  // dbe = Service<DQMStore>().operator->();

  // dbe->setCurrentFolder("CSC/EventInfo/DAQContents");
  
  ibooker.cd();
  ibooker.setCurrentFolder("CSC/EventInfo/DAQContents");
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

  ibooker.cd();
  ibooker.setCurrentFolder("CSC/EventInfo");
  mos.insert(std::make_pair("DAQSummary", ibooker.bookFloat("DAQSummary")));

  for (std::map<std::string, MonitorElement*>::iterator it = mos.begin(); it != mos.end(); it++) { 
    it->second->Fill(-1);
  }

}


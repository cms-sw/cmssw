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

#include "DQM/CSCMonitorModule/interface/CSCDaqInfo.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;
using namespace edm;

CSCDaqInfo::CSCDaqInfo(const edm::ParameterSet& ps) {
   
  FEDRange.first  = ps.getUntrackedParameter<unsigned int>("MinimumCSCFEDId", 750);
  FEDRange.second = ps.getUntrackedParameter<unsigned int>("MaximumCSCFEDId", 757);
  NumberOfFeds =FEDRange.second -  FEDRange.first + 1;

}

void CSCDaqInfo::beginJob(const edm::EventSetup& iSetup){

  dbe = 0;
  dbe = Service<DQMStore>().operator->();
       
  dbe->setCurrentFolder("CSC/EventInfo/DAQContents");
  DaqFraction = dbe->bookFloat("CSCDaqFraction");  
}

void CSCDaqInfo::beginLuminosityBlock(const LuminosityBlock& lumiBlock, const EventSetup& iSetup){
    
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

  if(0 != iSetup.find(recordKey) ) {

    //get fed summary information
    ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
    vector<int> FedsInIds= sumFED->m_fed_in;   
    int FedCount=0;

    //loop on all active feds
    for(unsigned int fedItr = 0; fedItr < FedsInIds.size(); ++fedItr) {
      int fedID = FedsInIds[fedItr];
      //make sure fed id is in allowed range  
      if (fedID >= FEDRange.first && fedID <= FEDRange.second) ++FedCount;
    }   

    //Fill active fed fraction ME
    if(NumberOfFeds > 0) {
      DaqFraction->Fill(FedCount / NumberOfFeds);
    } else {
      DaqFraction->Fill(-1);
    }

  } else {
    DaqFraction->Fill(-1);               
  }

}

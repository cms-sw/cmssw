/*
 * \file DTDataIntegrityTest.cc
 * 
 * \author S. Bolognesi - CERN
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 *
 */

#include <DQM/DTMonitorClient/src/DTDataIntegrityTest.h>

//Framework
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <string>


using namespace std;
using namespace edm;


DTDataIntegrityTest::DTDataIntegrityTest(const ParameterSet& ps) : nevents(0) {
  
  LogTrace ("DTDQM|DTRawToDigi|DTMonitorClient|DTDataIntegrityTest") << "[DTDataIntegrityTest]: Constructor";

  // prescale on the # of LS to update the test
  prescaleFactor = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);
 
  bookingdone = 0;
 
}


DTDataIntegrityTest::~DTDataIntegrityTest(){

  LogTrace ("DTDQM|DTRawToDigi|DTMonitorClient|DTDataIntegrityTest") << "DataIntegrityTest: analyzed " << nupdates << " updates";

}

  void DTDataIntegrityTest::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
                                          edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & context) {


  if (!bookingdone) {
  
  //nSTAEvents = 0;
  nupdates = 0;
  run=0;
  
  // book the summary histogram

  ibooker.setCurrentFolder("DT/00-DataIntegrity");

  summaryHisto = ibooker.book2D("DataIntegritySummary","Summary Data Integrity",12,1,13,5,-2,3);
  summaryHisto->setAxisTitle("Sector",1);
  summaryHisto->setAxisTitle("Wheel",2);

  ibooker.setCurrentFolder("DT/00-DataIntegrity");

  summaryTDCHisto = ibooker.book2D("DataIntegrityTDCSummary","TDC Summary Data Integrity",12,1,13,5,-2,3);
  summaryTDCHisto->setAxisTitle("Sector",1);
  summaryTDCHisto->setAxisTitle("Wheel",2);

  ibooker.setCurrentFolder("DT/00-DataIntegrity");

  glbSummaryHisto = ibooker.book2D("DataIntegrityGlbSummary","Summary Data Integrity",12,1,13,5,-2,3);
  glbSummaryHisto->setAxisTitle("Sector",1);
  glbSummaryHisto->setAxisTitle("Wheel",2);

  context.get<DTReadOutMappingRcd>().get(mapping);

  }
  bookingdone = 1; 


  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
  stringstream nLumiSegs_s; nLumiSegs_s << nLumiSegs;
  
  // prescale factor
  if (nLumiSegs%prescaleFactor != 0) return;
  
  LogTrace ("DTDQM|DTRawToDigi|DTMonitorClient|DTDataIntegrityTest")
    <<"[DTDataIntegrityTest]: End of LS " << nLumiSegs << ", performing client operations";


  // counts number of updats 
  nupdates++;

  
  //Counter for x bin in the timing histos
  counter++;

  //Loop on FED id
  for (int dduId=FEDNumbering::MINDTFEDID; dduId<=FEDNumbering::MAXDTFEDID; ++dduId){
    LogTrace ("DTDQM|DTRawToDigi|DTMonitorClient|DTDataIntegrityTest")
      <<"[DTDataIntegrityTest]:FED Id: "<<dduId;
 
    //Each nTimeBin onUpdate remove timing histos and book a new bunch of them
    stringstream dduId_s; dduId_s << dduId;
    
    string histoType;
    
    //Check if the list of ROS is compatible with the channels enabled
    string rosStatusName = "DT/00-DataIntegrity/FED" + dduId_s.str() + "/FED" + dduId_s.str() + "_ROSStatus";
    MonitorElement * FED_ROSStatus = igetter.get(rosStatusName);
     
    // Get the error summary histo
    string fedSummaryName = "DT/00-DataIntegrity/FED" + dduId_s.str() + "_ROSSummary";
    MonitorElement * FED_ROSSummary = igetter.get(fedSummaryName);

    // Get the event lenght plot (used to counr # of processed evts)
    string fedEvLenName = "DT/00-DataIntegrity/FED" + dduId_s.str() + "/FED" + dduId_s.str() + "_EventLenght";
    MonitorElement * FED_EvLenght = igetter.get(fedEvLenName);

    // Get the histos for FED integrity
    string fedIntegrityFolder = "DT/FEDIntegrity/";
    MonitorElement * hFEDEntry = igetter.get(fedIntegrityFolder+"FEDEntries");
    MonitorElement * hFEDFatal = igetter.get(fedIntegrityFolder+"FEDFatal");
    MonitorElement * hFEDNonFatal = igetter.get(fedIntegrityFolder+"FEDNonFatal");

    if(hFEDEntry && hFEDFatal && hFEDNonFatal) {

      if(FED_ROSSummary && FED_ROSStatus && FED_EvLenght) {
	TH2F * histoFEDSummary = FED_ROSSummary->getTH2F();
	TH2F * histoROSStatus  = FED_ROSStatus->getTH2F();
	TH1F * histoEvLenght   = FED_EvLenght->getTH1F();
	// Check that the FED is in the ReadOut using the FEDIntegrity histos
	bool fedNotReadout = (hFEDEntry->getBinContent(dduId-769) == 0 &&
			      hFEDFatal->getBinContent(dduId-769) == 0 &&
			      hFEDNonFatal->getBinContent(dduId-769) == 0);
	int nFEDEvts = histoEvLenght->Integral();
	for(int rosNumber = 1; rosNumber <= 12; ++rosNumber) { // loop on the ROS
	  int wheelNumber, sectorNumber;
	  if (!readOutToGeometry(dduId,rosNumber,wheelNumber,sectorNumber)) {
	    int result = -2;
	    float nErrors  = histoFEDSummary->Integral(1,14,rosNumber,rosNumber);
	    nErrors += histoROSStatus->Integral(2,8,rosNumber,rosNumber);
	    if(nErrors == 0) { // no errors
	      result = 0;
	    } else { // there are errors
	      result = 2;
	    }
	    summaryHisto->setBinContent(sectorNumber,wheelNumber+3,result);
	    int tdcResult = -2;
	    float nTDCErrors = histoFEDSummary->Integral(15,15,rosNumber,rosNumber); 
	    if(nTDCErrors == 0) { // no errors
	      tdcResult = 0;
	    } else { // there are errors
	      tdcResult = 2;
	    }
	    summaryTDCHisto->setBinContent(sectorNumber,wheelNumber+3,tdcResult);
	    // FIXME: different errors should have different weights
	    float sectPerc = max((float)0., ((float)nFEDEvts-nErrors)/(float)nFEDEvts);
	    glbSummaryHisto->setBinContent(sectorNumber,wheelNumber+3,sectPerc);
	   
	    if(fedNotReadout) {
	      // no data in this FED: it is off
	      summaryHisto->setBinContent(sectorNumber,wheelNumber+3,1);
	      summaryTDCHisto->setBinContent(sectorNumber,wheelNumber+3,1);
	      glbSummaryHisto->setBinContent(sectorNumber,wheelNumber+3,0);
	    }
	  }
	}
      
      } else { // no data in this FED: it is off
	for(int rosNumber = 1; rosNumber <= 12; ++rosNumber) {
	  int wheelNumber, sectorNumber;
	  if (!readOutToGeometry(dduId,rosNumber,wheelNumber,sectorNumber)) {
	    summaryHisto->setBinContent(sectorNumber,wheelNumber+3,1);
	    summaryTDCHisto->setBinContent(sectorNumber,wheelNumber+3,1);
	    glbSummaryHisto->setBinContent(sectorNumber,wheelNumber+3,0);
	  } 
	}
      }

    }
    
  }
  
}

void DTDataIntegrityTest::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  LogTrace ("DTDQM|DTRawToDigi|DTMonitorClient|DTDataIntegrityTest") <<"[DTDataIntegrityTest] endjob called!";
}

string DTDataIntegrityTest::getMEName(string histoType, int FEDId){
  //Use the DDU name to find the ME
  stringstream dduID_s; dduID_s << FEDId;

  string folderName = "DT/00-DataIntegrity/FED" + dduID_s.str(); 

  string histoName = folderName + "/FED" + dduID_s.str() + "_" + histoType;
  return histoName;
}

void DTDataIntegrityTest::bookHistos(DQMStore::IBooker & ibooker, string histoType, int dduId){
  stringstream dduId_s; dduId_s << dduId;

  ibooker.setCurrentFolder("DT/00-DataIntegrity/FED" + dduId_s.str());
  string histoName;

}


int DTDataIntegrityTest::readOutToGeometry(int dduId, int ros, int& wheel, int& sector){

  int dummy;
  return mapping->readOutToGeometry(dduId,ros,2,2,2,wheel,dummy,sector,dummy,dummy,dummy);

}


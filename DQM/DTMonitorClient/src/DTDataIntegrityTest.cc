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

  checkUros = ps.getUntrackedParameter<bool>("checkUros",false);
 
  bookingdone = false;
 
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
  } //booking
  bookingdone = true; 


  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
  string nLumiSegs_s = to_string(nLumiSegs);
  
  // prescale factor
  if (nLumiSegs%prescaleFactor != 0) return;
  
  LogTrace ("DTDQM|DTRawToDigi|DTMonitorClient|DTDataIntegrityTest")
    <<"[DTDataIntegrityTest]: End of LS " << nLumiSegs << ", performing client operations";


  // counts number of updats 
  nupdates++;

  
  //Counter for x bin in the timing histos
  counter++;

  //Loop on FED id
  //Monitoring only real used FEDs
  int FEDIDmax=FEDNumbering::MAXDTFEDID;
  int FEDIDmin=FEDNumbering::MINDTFEDID;
  if (checkUros){
	FEDIDmin=FEDNumbering::MINDTUROSFEDID;
	FEDIDmax=FEDNumbering::MAXDTUROSFEDID;
	}

  for (int dduId=FEDIDmin; dduId<=FEDIDmax; ++dduId){
    LogTrace ("DTDQM|DTRawToDigi|DTMonitorClient|DTDataIntegrityTest")
      <<"[DTDataIntegrityTest]:FED Id: "<<dduId;
 
    //Each nTimeBin onUpdate remove timing histos and book a new bunch of them
    string dduId_s = to_string(dduId);
    
    string histoType;
    
    //Check if the list of ROS is compatible with the channels enabled
    string rosStatusName = "DT/00-DataIntegrity/FED" + dduId_s + "/FED" + dduId_s + "_ROSStatus";
    if (checkUros) rosStatusName = "DT/00-DataIntegrity/FED" + dduId_s + "/FED" + dduId_s + "_uROSStatus";
    MonitorElement * FED_ROSStatus = igetter.get(rosStatusName);
     
    // Get the error summary histo
    string fedSummaryName = "DT/00-DataIntegrity/FED" + dduId_s + "_ROSSummary";
    MonitorElement * FED_ROSSummary = nullptr;
    MonitorElement * FED_ROSSummary1 = nullptr;
    MonitorElement * FED_ROSSummary2 = nullptr;
    string fedSummaryName1 = "";
    string fedSummaryName2 = "";
    string sign = "-";
    if (checkUros) { 
		if(dduId==FEDIDmin || dduId==FEDIDmax){
			if(dduId==FEDIDmax) sign="";
			fedSummaryName2 = "DT/00-DataIntegrity/ROSSummary_W" + sign + "2";
                	fedSummaryName1 = "DT/00-DataIntegrity/ROSSummary_W" + sign + "1";
			FED_ROSSummary1 = igetter.get(fedSummaryName1);
			FED_ROSSummary2 = igetter.get(fedSummaryName2);
		}
		else {	fedSummaryName = "DT/00-DataIntegrity/ROSSummary_W0";
			FED_ROSSummary1 = igetter.get(fedSummaryName);
			FED_ROSSummary2 = igetter.get(fedSummaryName); //for wheel compatibility...
		     }
    }
    else FED_ROSSummary = igetter.get(fedSummaryName); //legacy case

    // Get the event length plot (used to count # of processed evts)
    string fedEvLenName = "DT/00-DataIntegrity/FED" + dduId_s + "/FED" + dduId_s + "_EventLength";
    MonitorElement * FED_EvLength = igetter.get(fedEvLenName);

    // Get the histos for FED integrity
    string fedIntegrityFolder = "DT/00-DataIntegrity/";
    MonitorElement * hFEDEntry = igetter.get(fedIntegrityFolder+"FEDEntries");
    MonitorElement * hFEDFatal = igetter.get(fedIntegrityFolder+"FEDFatal");
    MonitorElement * hFEDNonFatal = igetter.get(fedIntegrityFolder+"FEDNonFatal");

    if (checkUros){

    if (hFEDEntry) {
	int offsetFED=1368;
	// Check that the FED is in the ReadOut using the FEDIntegrity histos
	bool fedNotReadout = (hFEDEntry->getBinContent(dduId-offsetFED)) == 0;
	int wheel = dduId-offsetFED-2;
	if(FED_ROSSummary1 && FED_ROSSummary2 && FED_ROSStatus && FED_EvLength && !fedNotReadout ) {
       		TH2F * histoFEDSummary1 = FED_ROSSummary1->getTH2F();
		TH2F * histoFEDSummary2 = FED_ROSSummary2->getTH2F(); // same for central FED...
       		TH2F * histoROSStatus  = FED_ROSStatus->getTH2F();
       		TH1F * histoEvLength   = FED_EvLength->getTH1F();

	        int nFEDEvts = histoEvLength->Integral();
		//if(dduId==FEDIDmin || dduId==FEDIDmax) nFEDEvts = nFEDEvts/2.; // split in 2 for external FEDs
		if (!(nFEDEvts>0)) continue;
		int extraFEDevents1 = 0;
		int extraFEDevents2 = 0;

	        for(int urosNumber = 1; urosNumber <= DOCESLOTS; ++urosNumber) { // loop on the uROS
			string urosNumber_s = to_string(urosNumber);
			// Get the event length plot for this uROS (used to count # of processed evts)
			string fedUrosEvLenName = "DT/00-DataIntegrity/FED" + dduId_s + "/uROS" + urosNumber_s + 
			"/FED" + dduId_s + "_uROS" + urosNumber_s + "_EventLength";
			MonitorElement * FED_uROS_EvLength = igetter.get(fedUrosEvLenName);
			TH1F * histoUrosEvLength   = FED_uROS_EvLength->getTH1F();
			int nFEDEvtsUros = histoUrosEvLength->Integral();

			//station 4 sectors drievn by uROS 1 & 2
			if (urosNumber==1) {extraFEDevents1=nFEDEvtsUros; continue;}
                        if (urosNumber==2) {extraFEDevents2=nFEDEvtsUros; continue;}

			if(nFEDEvtsUros>0) { //this uROS is active
			nFEDEvtsUros = extraFEDevents1+extraFEDevents2+nFEDEvtsUros / 3.; // split in 3 ROS	
			float nGErrors  = histoROSStatus->Integral(1,12,urosNumber,urosNumber);//Only Global Errors, 
			//not possible to distinguish which ROS, so coumting them in the 3/12 ROSes

		 	int ros = getROS(urosNumber,0);	
			for (int iros=ros; iros<(ros+3); ++iros){
				// -1,0,+1 wheels
		        	float nROBErrors1 = histoFEDSummary1->Integral(1,5,iros,iros); //Errors and Not OK Flag
         			float nErrors1 = nROBErrors1+nGErrors;
            			float result1 =0.;
            			if(nFEDEvtsUros!=0)
                			result1 =  max((float)0., ((float)nFEDEvtsUros-nROBErrors1)/(float)nFEDEvtsUros);
            			summaryHisto->setBinContent(iros,wheel+3,result1);
            			int tdcResult1 = -2;
            			float nTDCErrors1 = histoFEDSummary1->Integral(6,6,iros,iros); //Only TDC fatal considered
            			if(nTDCErrors1 == 0) { // no errors
              				tdcResult1 = 0.5;
            			} else { // there are errors
              				tdcResult1 = 2.5;
            			}
		        	summaryTDCHisto->setBinContent(iros,wheel+3,tdcResult1);

				// FIXME: different errors should have different weights
                 		float sectPerc1 = 0.;
				if(nFEDEvtsUros!=0)
					sectPerc1 = max((float)0., ((float)nFEDEvtsUros-nErrors1)/(float)nFEDEvtsUros);
            			glbSummaryHisto->setBinContent(iros,wheel+3,sectPerc1);
				if(dduId==(FEDIDmax-1)) continue; //wheel 0 case

				// -2,+2 wheels
				float nROBErrors2 = histoFEDSummary2->Integral(1,5,iros,iros); //Errors and Not OK Flag
                                float nErrors2 = nROBErrors2+nGErrors;
                                float result2 =0.;
                                if(nFEDEvtsUros!=0)
                                        result2 =  max((float)0., ((float)nFEDEvtsUros-nROBErrors2)/(float)nFEDEvtsUros);
                                summaryHisto->setBinContent(iros,wheel*2+3,result2);

                                int tdcResult2 = -2;
                                float nTDCErrors2 = histoFEDSummary2->Integral(6,6,iros,iros); //Only TDC fatal considered
                                if(nTDCErrors2 == 0) { // no errors
                                        tdcResult2 = 0.5;
                                } else { // there are errors
                                        tdcResult2 = 2.5;
                                }
                                summaryTDCHisto->setBinContent(iros,wheel*2+3,tdcResult2);

                                // FIXME: different errors should have different weights
                                float sectPerc2 = 0.;
                                if(nFEDEvtsUros!=0)
                                        sectPerc2 = max((float)0., ((float)nFEDEvtsUros-nErrors2)/(float)nFEDEvtsUros);
                                glbSummaryHisto->setBinContent(iros,wheel*2+3,sectPerc2);
			    } //loop in three ros
			}// this uROS is active		
 		  } //loop on uros
	   } else { // no data in this FED: it is off, no ROS suummary/status or evLength and fedNotReadout
		        for(int i = 1; i <= DOCESLOTS; ++i) {
            			summaryHisto->setBinContent(i,wheel+3,0.5);
            			summaryTDCHisto->setBinContent(i,wheel+3,1.5);
            			glbSummaryHisto->setBinContent(i,wheel+3,0.5);
				if (dduId==(FEDIDmax-1)) continue; //wheel 0 case
                                summaryHisto->setBinContent(i,wheel*2+3,0.5);
                                summaryTDCHisto->setBinContent(i,wheel*2+3,1.5);
                                glbSummaryHisto->setBinContent(i,wheel*2+3,0.5);
        		} //loop on uros
      	     } // no data in this FED: it is off, no ROS suummary/status or evLength

    } //FEDentry
    } else { //legacy case
    if(hFEDEntry && hFEDFatal && hFEDNonFatal)  {
      if(FED_ROSSummary && FED_ROSStatus && FED_EvLength) {
	TH2F * histoFEDSummary = FED_ROSSummary->getTH2F();
	TH2F * histoROSStatus  = FED_ROSStatus->getTH2F();
	TH1F * histoEvLength   = FED_EvLength->getTH1F();
	// Check that the FED is in the ReadOut using the FEDIntegrity histos
	bool fedNotReadout = (hFEDEntry->getBinContent(dduId-769) == 0 &&
			      hFEDFatal->getBinContent(dduId-769) == 0 &&
			      hFEDNonFatal->getBinContent(dduId-769) == 0);
		
	int nFEDEvts = histoEvLength->Integral();
	for(int rosNumber = 1; rosNumber <= 12; ++rosNumber) { // loop on the ROS
	  int wheelNumber, sectorNumber;
	  if (!readOutToGeometry(dduId,rosNumber,wheelNumber,sectorNumber)) {
	    float nErrors  = histoFEDSummary->Integral(1,14,rosNumber,rosNumber);
	    float nROBErrors = histoROSStatus->Integral(2,8,rosNumber,rosNumber);
	    nErrors += nROBErrors;
	    float result =0.;
	    if(nFEDEvts!=0) 
		result =  max((float)0., ((float)nFEDEvts-nROBErrors)/(float)nFEDEvts); 
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
	      summaryHisto->setBinContent(sectorNumber,wheelNumber+3,0);
	      summaryTDCHisto->setBinContent(sectorNumber,wheelNumber+3,1);
	      glbSummaryHisto->setBinContent(sectorNumber,wheelNumber+3,0);
	    } //fedNotReadout
	  } //mapping
	} //loop on ros
      } else { // no data in this FED: it is off, no ROS suummary/status or evLength
	for(int rosNumber = 1; rosNumber <= 12; ++rosNumber) {
	  int wheelNumber, sectorNumber;
	  if (!readOutToGeometry(dduId,rosNumber,wheelNumber,sectorNumber)) {
	    summaryHisto->setBinContent(sectorNumber,wheelNumber+3,0);
	    summaryTDCHisto->setBinContent(sectorNumber,wheelNumber+3,1);
	    glbSummaryHisto->setBinContent(sectorNumber,wheelNumber+3,0);
	  } //mapping
	} //loop on ros
      } // no data in this FED: it is off, no ROS suummary/status or evLength

    }// no FED entry, fatal, nonfatal
    
   } //legacy case

  } //  loop on dduIds
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

int DTDataIntegrityTest::getROS(int uROS, int link){

  int slot = 0;
  switch(uROS){
  case 1:{
	slot = 5;
	break;
  }
  case 2:{
	slot = 11;
	break;
  }
  case 3:{
	slot = 1;
	break;
  }
  case 4:{
	slot = 7;
	break;
  }
  case 5:{
	slot = 2;
	break;
  }
  case 6:{
	slot = 8;
	break;
  }
  case 9:{
	slot = 9;
	break;
  }
  case 10:{
	slot = 3;
	break;
  }
  case 11:{
	slot = 10;
	break;
  }
  case 12:{
	slot = 4;
	break;
  }
	}
	
  if (slot%6 == 5) return link+1;

  int ros = (link/24) + 3*(slot%6) - 2;
  return ros;
}

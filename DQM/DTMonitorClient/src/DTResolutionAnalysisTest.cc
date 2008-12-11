

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/11/05 17:43:16 $
 *  $Revision: 1.1 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTResolutionAnalysisTest.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <sstream>
#include <math.h>


using namespace edm;
using namespace std;


DTResolutionAnalysisTest::DTResolutionAnalysisTest(const edm::ParameterSet& ps){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "[DTResolutionAnalysisTest]: Constructor";

  dbe = edm::Service<DQMStore>().operator->();

  prescaleFactor = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);

  // quality test names
  MeanCriterionName = ps.getUntrackedParameter<string>("meanTestName","ResidualsMeanInRange"); 
  SigmaCriterionName = ps.getUntrackedParameter<string>("sigmaTestName","ResidualsSigmaInRange"); 
  
 }


DTResolutionAnalysisTest::~DTResolutionAnalysisTest(){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "DTResolutionAnalysisTest: analyzed " << nevents << " events";

}


void DTResolutionAnalysisTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") <<"[DTResolutionAnalysisTest]: BeginJob"; 

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // loop over all the CMS wheel & book the summary histos
  for (int wheel=-2; wheel<=2; wheel++){
    bookHistos(wheel);
  }

}


void DTResolutionAnalysisTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") <<"[DTResolutionAnalysisTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

}


void DTResolutionAnalysisTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "[DTResolutionAnalysisTest]: "<<nevents<<" events";

}



void DTResolutionAnalysisTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") <<"[DTResolutionAnalysisTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
  
  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  for(int wheel=-2; wheel<=2; wheel++){
    for(int sector=1; sector<=14; sector++){

      stringstream wh; wh << wheel;
      stringstream sect; sect << sector;

      // Mean test 

      // get the histo
      string name= "MeanTest_W" + wh.str() + "_Sec" + sect.str();
      string theHisto= "DT/02-Segments/Wheel" +  wh.str() + "/Tests/" + name;
      MonitorElement * resMean_histo = dbe->get(theHisto);
      if(resMean_histo){
		
	const QReport * theMeanQReport = resMean_histo->getQReport(MeanCriterionName);
	// Report the channels failing the test on the mean
	if(theMeanQReport) { 
 
	  vector<dqm::me_util::Channel> badChannels = theMeanQReport->getBadChannels();
	  for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	       channel != badChannels.end(); channel++) {
	    edm::LogError("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "Bad mean channel: wh: " << wheel
				     << " st: " << stationFromBin((*channel).getBin())
				     << " sect: " <<sector
				     << " sl: " << slFromBin((*channel).getBin())
				     << " mean (cm): " << (*channel).getContents();
	    // fill the wheel summary histos
	    wheelMeanHistos[wheel]->Fill(sector,(*channel).getBin());
	    wheelMeanHistos[3]->Fill(sector,wheel);
	  }
	}
      }
      
      // Sigma test

      // get the histo
      name= "SigmaTest_W" + wh.str() + "_Sec" + sect.str();
      theHisto= "DT/02-Segments/Wheel" +  wh.str() + "/Tests/" + name;
      MonitorElement * resSigma_histo = dbe->get(theHisto);
      if(resSigma_histo){
       
	// Report the channels failing the test on the sigma
	const QReport * theSigmaQReport = resSigma_histo->getQReport(SigmaCriterionName);
	if(theSigmaQReport) {
	  vector<dqm::me_util::Channel> badChannels = theSigmaQReport->getBadChannels();
	  for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	       channel != badChannels.end(); channel++) {
	    edm::LogError("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "Bad sigma: wh: " << wheel
				     << " st: " << stationFromBin((*channel).getBin())
				     << " sect: " <<sector
				     << " sl: " << slFromBin((*channel).getBin())
				     << " sigma (cm): " << (*channel).getContents();
	    // fill the wheel summary histos
	    wheelSigmaHistos[wheel]->Fill(sector,(*channel).getBin());
	    wheelSigmaHistos[3]->Fill(sector,wheel);
	  }
	}
      }

    } // loop over sectors
  } // loop over wheels

}



void DTResolutionAnalysisTest::endJob(){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "[DTResolutionAnalysisTest] endjob called!";
}


void DTResolutionAnalysisTest::bookHistos(int wh) {
  
  dbe->setCurrentFolder("DT/05-Residuals");

  stringstream wheel; wheel <<wh;
  
  string histoName =  "MeanSummaryRes_testFailed_W" + wheel.str();
  wheelMeanHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,1,15,11,1,12);
  wheelMeanHistos[wh]->setBinLabel(1,"Sector1",1);
  wheelMeanHistos[wh]->setBinLabel(2,"Sector2",1);
  wheelMeanHistos[wh]->setBinLabel(3,"Sector3",1);
  wheelMeanHistos[wh]->setBinLabel(4,"Sector4",1);
  wheelMeanHistos[wh]->setBinLabel(5,"Sector5",1);
  wheelMeanHistos[wh]->setBinLabel(6,"Sector6",1);
  wheelMeanHistos[wh]->setBinLabel(7,"Sector7",1);
  wheelMeanHistos[wh]->setBinLabel(8,"Sector8",1);
  wheelMeanHistos[wh]->setBinLabel(9,"Sector9",1);
  wheelMeanHistos[wh]->setBinLabel(10,"Sector10",1);
  wheelMeanHistos[wh]->setBinLabel(11,"Sector11",1);
  wheelMeanHistos[wh]->setBinLabel(12,"Sector12",1);
  wheelMeanHistos[wh]->setBinLabel(13,"Sector13",1);
  wheelMeanHistos[wh]->setBinLabel(14,"Sector14",1);
  wheelMeanHistos[wh]->setBinLabel(1,"MB1_SL1",2);
  wheelMeanHistos[wh]->setBinLabel(2,"MB1_SL2",2);
  wheelMeanHistos[wh]->setBinLabel(3,"MB1_SL3",2);
  wheelMeanHistos[wh]->setBinLabel(4,"MB2_SL1",2);
  wheelMeanHistos[wh]->setBinLabel(5,"MB2_SL2",2);
  wheelMeanHistos[wh]->setBinLabel(6,"MB2_SL3",2);
  wheelMeanHistos[wh]->setBinLabel(7,"MB3_SL1",2);
  wheelMeanHistos[wh]->setBinLabel(8,"MB3_SL2",2);
  wheelMeanHistos[wh]->setBinLabel(9,"MB3_SL3",2);
  wheelMeanHistos[wh]->setBinLabel(10,"MB4_SL1",2);
  wheelMeanHistos[wh]->setBinLabel(11,"MB4_SL3",2); 
  
  histoName =  "SigmaSummaryRes_testFailed_W" + wheel.str();
  wheelSigmaHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,1,15,11,1,12);
  wheelSigmaHistos[wh]->setBinLabel(1,"Sector1",1);
  wheelSigmaHistos[wh]->setBinLabel(2,"Sector2",1);
  wheelSigmaHistos[wh]->setBinLabel(3,"Sector3",1);
  wheelSigmaHistos[wh]->setBinLabel(4,"Sector4",1);
  wheelSigmaHistos[wh]->setBinLabel(5,"Sector5",1);
  wheelSigmaHistos[wh]->setBinLabel(6,"Sector6",1);
  wheelSigmaHistos[wh]->setBinLabel(7,"Sector7",1);
  wheelSigmaHistos[wh]->setBinLabel(8,"Sector8",1);
  wheelSigmaHistos[wh]->setBinLabel(9,"Sector9",1);
  wheelSigmaHistos[wh]->setBinLabel(10,"Sector10",1);
  wheelSigmaHistos[wh]->setBinLabel(11,"Sector11",1);
  wheelSigmaHistos[wh]->setBinLabel(12,"Sector12",1);
  wheelSigmaHistos[wh]->setBinLabel(13,"Sector13",1);
  wheelSigmaHistos[wh]->setBinLabel(14,"Sector14",1);
  wheelSigmaHistos[wh]->setBinLabel(1,"MB1_SL1",2);
  wheelSigmaHistos[wh]->setBinLabel(2,"MB1_SL2",2);
  wheelSigmaHistos[wh]->setBinLabel(3,"MB1_SL3",2);
  wheelSigmaHistos[wh]->setBinLabel(4,"MB2_SL1",2);
  wheelSigmaHistos[wh]->setBinLabel(5,"MB2_SL2",2);
  wheelSigmaHistos[wh]->setBinLabel(6,"MB2_SL3",2);
  wheelSigmaHistos[wh]->setBinLabel(7,"MB3_SL1",2);
  wheelSigmaHistos[wh]->setBinLabel(8,"MB3_SL2",2);
  wheelSigmaHistos[wh]->setBinLabel(9,"MB3_SL3",2);
  wheelSigmaHistos[wh]->setBinLabel(10,"MB4_SL1",2);
  wheelSigmaHistos[wh]->setBinLabel(11,"MB4_SL3",2);
  
  if(wheelMeanHistos.find(3) == wheelMeanHistos.end()){
    string histoName =  "MeanSummaryRes_testFailed";
    wheelMeanHistos[3] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,1,15,5,-2,3);
    wheelMeanHistos[3]->setBinLabel(1,"Sector1",1);
    wheelMeanHistos[3]->setBinLabel(1,"Sector1",1);
    wheelMeanHistos[3]->setBinLabel(2,"Sector2",1);
    wheelMeanHistos[3]->setBinLabel(3,"Sector3",1);
    wheelMeanHistos[3]->setBinLabel(4,"Sector4",1);
    wheelMeanHistos[3]->setBinLabel(5,"Sector5",1);
    wheelMeanHistos[3]->setBinLabel(6,"Sector6",1);
    wheelMeanHistos[3]->setBinLabel(7,"Sector7",1);
    wheelMeanHistos[3]->setBinLabel(8,"Sector8",1);
    wheelMeanHistos[3]->setBinLabel(9,"Sector9",1);
    wheelMeanHistos[3]->setBinLabel(10,"Sector10",1);
    wheelMeanHistos[3]->setBinLabel(11,"Sector11",1);
    wheelMeanHistos[3]->setBinLabel(12,"Sector12",1);
    wheelMeanHistos[3]->setBinLabel(13,"Sector13",1);
    wheelMeanHistos[3]->setBinLabel(14,"Sector14",1);
    wheelMeanHistos[3]->setBinLabel(1,"Wheel-2",2);
    wheelMeanHistos[3]->setBinLabel(2,"Wheel-1",2);
    wheelMeanHistos[3]->setBinLabel(3,"Wheel0",2);
    wheelMeanHistos[3]->setBinLabel(4,"Wheel+1",2);
    wheelMeanHistos[3]->setBinLabel(5,"Wheel+2",2);
  }

  if(wheelSigmaHistos.find(3) == wheelSigmaHistos.end()){
    string histoName =  "SigmaSummaryRes_testFailed";
    wheelSigmaHistos[3] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,1,15,5,-2,3);
    wheelSigmaHistos[3]->setBinLabel(1,"Sector1",1);
    wheelSigmaHistos[3]->setBinLabel(1,"Sector1",1);
    wheelSigmaHistos[3]->setBinLabel(2,"Sector2",1);
    wheelSigmaHistos[3]->setBinLabel(3,"Sector3",1);
    wheelSigmaHistos[3]->setBinLabel(4,"Sector4",1);
    wheelSigmaHistos[3]->setBinLabel(5,"Sector5",1);
    wheelSigmaHistos[3]->setBinLabel(6,"Sector6",1);
    wheelSigmaHistos[3]->setBinLabel(7,"Sector7",1);
    wheelSigmaHistos[3]->setBinLabel(8,"Sector8",1);
    wheelSigmaHistos[3]->setBinLabel(9,"Sector9",1);
    wheelSigmaHistos[3]->setBinLabel(10,"Sector10",1);
    wheelSigmaHistos[3]->setBinLabel(11,"Sector11",1);
    wheelSigmaHistos[3]->setBinLabel(12,"Sector12",1);
    wheelSigmaHistos[3]->setBinLabel(13,"Sector13",1);
    wheelSigmaHistos[3]->setBinLabel(14,"Sector14",1);
    wheelSigmaHistos[3]->setBinLabel(1,"Wheel-2",2);
    wheelSigmaHistos[3]->setBinLabel(2,"Wheel-1",2);
    wheelSigmaHistos[3]->setBinLabel(3,"Wheel0",2);
    wheelSigmaHistos[3]->setBinLabel(4,"Wheel+1",2);
    wheelSigmaHistos[3]->setBinLabel(5,"Wheel+2",2);
  }  
}


int DTResolutionAnalysisTest::stationFromBin(int bin) const {
  return (int) (bin /3.1)+1;
}
 

int DTResolutionAnalysisTest::slFromBin(int bin) const {
  int ret = bin%3;
  if(ret == 0 || bin == 11) ret = 3;
  
  return ret;
}

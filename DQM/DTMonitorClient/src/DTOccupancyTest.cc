/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - University and INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 */


#include <DQM/DTMonitorClient/src/DTOccupancyTest.h>
#include <DQM/DTMonitorClient/src/DTOccupancyClusterBuilder.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMath.h"

using namespace edm;
using namespace std;




DTOccupancyTest::DTOccupancyTest(const edm::ParameterSet& ps){
  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTest") << "[DTOccupancyTest]: Constructor";

  // Get the DQM service

  lsCounter = 0;

  writeRootFile  = ps.getUntrackedParameter<bool>("writeRootFile", false);
  if(writeRootFile) {
    rootFile = new TFile("DTOccupancyTest.root","RECREATE");
    ntuple = new TNtuple("OccupancyNtuple", "OccupancyNtuple", "ls:wh:st:se:lay1MeanCell:lay1RMS:lay2MeanCell:lay2RMS:lay3MeanCell:lay3RMS:lay4MeanCell:lay4RMS:lay5MeanCell:lay5RMS:lay6MeanCell:lay6RMS:lay7MeanCell:lay7RMS:lay8MeanCell:lay8RMS:lay9MeanCell:lay9RMS:lay10MeanCell:lay10RMS:lay11MeanCell:lay11RMS:lay12MeanCell:lay12RMS");
  }
  
  // switch on the mode for running on test pulses (different top folder)
  tpMode = ps.getUntrackedParameter<bool>("testPulseMode", false);
  
  runOnAllHitsOccupancies =  ps.getUntrackedParameter<bool>("runOnAllHitsOccupancies", true);
  runOnNoiseOccupancies =  ps.getUntrackedParameter<bool>("runOnNoiseOccupancies", false);
  runOnInTimeOccupancies = ps.getUntrackedParameter<bool>("runOnInTimeOccupancies", false);
  nMinEvts  = ps.getUntrackedParameter<int>("nEventsCert", 5000);

  bookingdone = 0;

  // Event counter
  nevents = 0;

}

DTOccupancyTest::~DTOccupancyTest(){
  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTest") << " destructor called" << endl;
}


void DTOccupancyTest::beginRun(const edm::Run& run, const EventSetup& context){

  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTest") << "[DTOccupancyTest]: BeginRun";

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}

  void DTOccupancyTest::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
                                edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & context) {
  if (!bookingdone) {

  // Book the summary histos
  //   - one summary per wheel
  for(int wh = -2; wh <= 2; ++wh) { // loop over wheels
    bookHistos(ibooker,wh, string("Occupancies"), "OccupancySummary");
    }

  ibooker.setCurrentFolder(topFolder());
  string title = "Occupancy Summary";
  if(tpMode) {
    title = "Test Pulse Occupancy Summary";
    }
  //   - global summary with alarms
  summaryHisto = ibooker.book2D("OccupancySummary",title.c_str(),12,1,13,5,-2,3);
  summaryHisto->setAxisTitle("sector",1);
  summaryHisto->setAxisTitle("wheel",2);
  
  //   - global summary with percentages
  glbSummaryHisto = ibooker.book2D("OccupancyGlbSummary",title.c_str(),12,1,13,5,-2,3);
  glbSummaryHisto->setAxisTitle("sector",1);
  glbSummaryHisto->setAxisTitle("wheel",2);


  // assign the name of the input histogram
  if(runOnAllHitsOccupancies) {
    nameMonitoredHisto = "OccupancyAllHits_perCh";
    } else if(runOnNoiseOccupancies) {
    nameMonitoredHisto = "OccupancyNoise_perCh";
    } else if(runOnInTimeOccupancies) {
    nameMonitoredHisto = "OccupancyInTimeHits_perCh";
    } else { // default is AllHits histo
    nameMonitoredHisto = "OccupancyAllHits_perCh";
    }

  }
  bookingdone = 1; 


  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTest")
    <<"[DTOccupancyTest]: End of LS transition, performing the DQM client operation";
  lsCounter++;



  // Reset the global summary
  summaryHisto->Reset();
  glbSummaryHisto->Reset();

  // Get all the DT chambers
  vector<const DTChamber*> chambers = muonGeom->chambers();

  for(vector<const DTChamber*>::const_iterator chamber = chambers.begin();
      chamber != chambers.end(); ++chamber) {  // Loop over all chambers
    DTChamberId chId = (*chamber)->id();

    MonitorElement * chamberOccupancyHisto = igetter.get(getMEName(nameMonitoredHisto, chId));	

    // Run the tests on the plot for the various granularities
    if(chamberOccupancyHisto != 0) {
      // Get the 2D histo
      TH2F* histo = chamberOccupancyHisto->getTH2F();
      float chamberPercentage = 1.;
      int result = runOccupancyTest(histo, chId, chamberPercentage);
      int sector = chId.sector();

      if(sector == 13) {
	sector = 4;
	float resultSect4 = wheelHistos[chId.wheel()]->getBinContent(sector, chId.station());
	if(resultSect4 > result) {
	  result = (int)resultSect4;
	}
      } else if(sector == 14) {
	sector = 10;
	float resultSect10 = wheelHistos[chId.wheel()]->getBinContent(sector, chId.station());
	if(resultSect10 > result) {
	  result = (int)resultSect10;
	}
      }
      
      // the 2 MB4 of Sect 4 and 10 count as half a chamber
      if((sector == 4 || sector == 10) && chId.station() == 4) 
	chamberPercentage = chamberPercentage/2.;

      wheelHistos[chId.wheel()]->setBinContent(sector, chId.station(),result);
      if(result > summaryHisto->getBinContent(sector, chId.wheel()+3)) {
	summaryHisto->setBinContent(sector, chId.wheel()+3, result);
      }
      glbSummaryHisto->Fill(sector, chId.wheel(), chamberPercentage*1./4.);
    } else {
      LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTest") << "[DTOccupancyTest] ME: "
				      << getMEName(nameMonitoredHisto, chId) << " not found!" << endl;
    }

  }

  string nEvtsName = "DT/EventInfo/Counters/nProcessedEventsDigi";

  MonitorElement * meProcEvts = igetter.get(nEvtsName);

  if (meProcEvts) {
    int nProcEvts = meProcEvts->getFloatValue();
    glbSummaryHisto->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
    summaryHisto->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
  } else {
    glbSummaryHisto->setEntries(nMinEvts +1);
    summaryHisto->setEntries(nMinEvts + 1);
    LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTest") << "[DTOccupancyTest] ME: "
		       <<  nEvtsName << " not found!" << endl;
  }

  // Fill the global summary
  // Check for entire sectors off and report them on the global summary
  //FIXME: TODO

  if(writeRootFile) ntuple->AutoSave("SaveSelf");

}

void DTOccupancyTest::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTest") << "[DTOccupancyTest] endjob called!";
  if(writeRootFile) {
    rootFile->cd();
    ntuple->Write();
    rootFile->Close();
  }
}


  
// --------------------------------------------------

void DTOccupancyTest::bookHistos(DQMStore::IBooker & ibooker, const int wheelId, 
                                                      string folder, string histoTag) {
  // Set the current folder
  stringstream wheel; wheel << wheelId;	

  ibooker.setCurrentFolder(topFolder());

  // build the histo name
  string histoName = histoTag + "_W" + wheel.str(); 
  
  
  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTest") <<"[DTOccupancyTest]: booking wheel histo:"
							<< histoName 
							<< " (tag "
							<< histoTag
							<< ") in: "
							<< topFolder() + "Wheel"+ wheel.str() + "/" + folder << endl;
  
  string histoTitle = "Occupancy summary WHEEL: "+wheel.str();
  if(tpMode) {
    histoTitle = "TP Occupancy summary WHEEL: "+wheel.str();
  }

  wheelHistos[wheelId] = ibooker.book2D(histoName,histoTitle,12,1,13,4,1,5);
  wheelHistos[wheelId]->setBinLabel(1,"MB1",2);
  wheelHistos[wheelId]->setBinLabel(2,"MB2",2);
  wheelHistos[wheelId]->setBinLabel(3,"MB3",2);
  wheelHistos[wheelId]->setBinLabel(4,"MB4",2);
  wheelHistos[wheelId]->setAxisTitle("sector",1);
}



string DTOccupancyTest::getMEName(string histoTag, const DTChamberId& chId) {

  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();
  stringstream sector; sector << chId.sector();


  string folderRoot = topFolder() + "Wheel" + wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/";

  string folder = "Occupancies/";
  
  // build the histo name
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str();

  string histoname = folderRoot + histoName;

  return histoname;
}




// Run a test on the occupancy of the chamber
// Return values:
// 0 -> all ok
// 1 -> # consecutive dead channels > N
// 2 -> dead layer
// 3 -> dead SL
// 4 -> dead chamber
int DTOccupancyTest::runOccupancyTest(TH2F *histo, const DTChamberId& chId,
				      float& chamberPercentage) {
  int nBinsX = histo->GetNbinsX();

  // Reset the error flags
  bool failSL = false;
  bool failLayer = false;
  bool failCells = false;

  // Check that the chamber has digis
  if(histo->Integral() == 0) {
    chamberPercentage = 0;
    return 4;
  }

  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "--- Occupancy test for chamber: " << chId << endl;
  // set the # of SLs
  int nSL = 3;
  if(chId.station() == 4) nSL = 2;

  // 
  float values[28];
  if(writeRootFile) {
    values[0] = lsCounter;
    values[1] = chId.wheel(); 
    values[2] = chId.station();
    values[3] = chId.sector();
  }

  // Compute the average occupancy per layer and its RMS
  // we also look of the layer with the smallest RMS in order to find a reference value
  // for the cell occupancy 
  double totalChamberOccupp = 0;
  double squaredLayerOccupSum = 0;

  map<DTLayerId, pair<double, double> > averageCellOccupAndRMS;
  map<DTLayerId, double> layerOccupancyMap;

  int index = 3;
  for(int slay = 1; slay <= 3; ++slay) { // loop over SLs
    // Skip layer 2 on MB4
    if(chId.station() == 4 && slay == 2) {
      if(writeRootFile) {
	values[12] = -1;
	values[13] = -1; 
	values[14] = -1;
	values[15] = -1;
	values[16] = -1;
	values[17] = -1; 
	values[18] = -1;
	values[19] = -1;
      }
      index = 19;
      continue;
    }
    // check the SL occupancy
    int binYlow = ((slay-1)*4)+1;
    int binYhigh = binYlow+3;
    double slInteg = histo->Integral(1,nBinsX,binYlow,binYhigh);
    if(slInteg == 0) {
      chamberPercentage = 1.-1./(float)nSL;
      return 3;
    }

    for(int lay = 1; lay <= 4; ++lay) { // loop over layers
      DTLayerId layID(chId,slay,lay);

      int binY = binYlow+(lay-1);
      
      double layerInteg = histo->Integral(1,nBinsX,binY,binY);
      squaredLayerOccupSum += layerInteg*layerInteg;
      totalChamberOccupp+= layerInteg;

      layerOccupancyMap[layID] = layerInteg;

      // We look for the distribution of hits within the layer
      int nWires = muonGeom->layer(layID)->specificTopology().channels();
      int firstWire = muonGeom->layer(layID)->specificTopology().firstChannel();
      double layerSquaredSum = 0;
      // reset the alert bit in the plot (used by render plugins)
      histo->SetBinContent(nBinsX+1,binY,0.);

      for(int cell = firstWire; cell != (nWires+firstWire); ++cell) { // loop over cells
	double cellOccup = histo->GetBinContent(cell,binY);
	layerSquaredSum+=cellOccup*cellOccup;
      }
      


      // compute the average cell occpuancy and RMS
      double averageCellOccup = layerInteg/nWires;
      double averageSquaredCellOccup = layerSquaredSum/nWires;
      double rmsCellOccup = sqrt(averageSquaredCellOccup - averageCellOccup*averageCellOccup);
      averageCellOccupAndRMS[layID] = make_pair(averageCellOccup, rmsCellOccup);
      LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "  " << layID
							<< " average cell occ.: " << averageCellOccup
							<< " RMS: " << rmsCellOccup << endl;
      if(writeRootFile) {
	index++;
	values[index] = averageCellOccup;
	index++;
	values[index] = rmsCellOccup;
      }
    }
  }
  

  if(writeRootFile) ntuple->Fill(values);

  double minCellRMS = 99999999;
  double referenceCellOccup = -1;

  DTOccupancyClusterBuilder builder;

  // find the cell reference value
  for(map<DTLayerId, pair<double, double> >::const_iterator layAndValues = averageCellOccupAndRMS.begin();
      layAndValues != averageCellOccupAndRMS.end(); layAndValues++) {
    DTLayerId lid = (*layAndValues).first;

    double rms = (*layAndValues).second.second;
    double lOcc = layerOccupancyMap[lid]; // FIXME: useless
    double avCellOcc = (*layAndValues).second.first;
    LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "   " << lid << " tot. occ: " << lOcc
						      << " average cell occ: " << avCellOcc
						      << " RMS: " << rms << endl;

    if(avCellOcc != 0) {
      DTOccupancyPoint point(avCellOcc, rms, lid);
      builder.addPoint(point);
    } else {
      if(monitoredLayers.find(lid) == monitoredLayers.end()) monitoredLayers.insert(lid);
    }
  }

  builder.buildClusters();
  referenceCellOccup = builder.getBestCluster().averageMean();
  minCellRMS = builder.getBestCluster().averageRMS();

  double safeFactor = 3.;

  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << " Reference cell occup.: " << referenceCellOccup
						    << " RMS: " << minCellRMS << endl;
  
  int nFailingSLs = 0;

  // Check the layer occupancy
  for(int slay = 1; slay <= 3; ++slay) { // loop over SLs
    // Skip layer 2 on MB4
    if(chId.station() == 4 && slay == 2) continue;

    int binYlow = ((slay-1)*4)+1;
    int nFailingLayers = 0;

    for(int lay = 1; lay <= 4; ++lay) { // loop over layers
      DTLayerId layID(chId,slay,lay);
      int nWires = muonGeom->layer(layID)->specificTopology().channels();
      int firstWire = muonGeom->layer(layID)->specificTopology().firstChannel();
      int binY = binYlow+(lay-1);

      // compute the integral of the layer occupancy
      double layerInteg = histo->Integral(1,nBinsX,binY,binY);

      LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "     layer: " << layID << " integral: " << layerInteg << endl;

      // Check if in the list of layers which are monitored
      bool alreadyMonitored = false;
      if(monitoredLayers.find(layID) != monitoredLayers.end()) alreadyMonitored = true;


      if(layerInteg == 0) { // layer is dead (no need to go further
	LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "     fail layer: no entries" << endl;
	// Add it to the list of of monitored layers
	if(!alreadyMonitored) monitoredLayers.insert(layID);
	nFailingLayers++;
	failLayer = true;
	histo->SetBinContent(nBinsX+1,binY,-1.);
	// go to next layer
	continue;
      }

	if(alreadyMonitored || builder.isProblematic(layID)) { // check the layer

	  // Add it to the list of of monitored layers
	  if(monitoredLayers.find(layID) == monitoredLayers.end()) monitoredLayers.insert(layID);

	  int totalDeadCells = 0;
	  int nDeadCellsInARow = 1;
	  int nDeadCellsInARowMax = 0;
	  int nCellsZeroCount = 0;
	  bool previousIsDead = false;

	  int interDeadCells = 0;
	  for(int cell = firstWire; cell != (nWires+firstWire); ++cell) { // loop over cells
	    double cellOccup = histo->GetBinContent(cell,binY);
	    LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "       cell occup: " << cellOccup;
	    if(cellOccup == 0 || cellOccup < (referenceCellOccup-safeFactor*sqrt(referenceCellOccup))) {
	      if(cellOccup == 0) nCellsZeroCount++;
	      totalDeadCells++;
	      if(previousIsDead) nDeadCellsInARow++;
	      previousIsDead = true;
	      interDeadCells = 0;
	      LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "       below reference" << endl;
	    } else {
	      previousIsDead = false;
	      interDeadCells++;

	      // 3 cells not dead between a group of dead cells don't break the count
	      if(interDeadCells > 3) {
		if(nDeadCellsInARow > nDeadCellsInARowMax) nDeadCellsInARowMax = nDeadCellsInARow;
		nDeadCellsInARow = 1; 
	      }
	    }
	  }
	  if(nDeadCellsInARow > nDeadCellsInARowMax) nDeadCellsInARowMax = nDeadCellsInARow;
	  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "       # wires: " << nWires
							    << " # cells 0 count: " << nCellsZeroCount
							    << " # dead cells in a row: " << nDeadCellsInARowMax
							    << " total # of dead cells: " << totalDeadCells;
	  
	  if((TMath::Erfc(referenceCellOccup/sqrt(referenceCellOccup)) < 10./(double)nWires &&
	      nDeadCellsInARowMax>= 10.) ||
	     (TMath::Erfc(referenceCellOccup/sqrt(referenceCellOccup)) < 0.5 &&
	      totalDeadCells > nWires/2.)) {
	    LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << " -> fail layer!" << endl;
	    nFailingLayers++;
	    failLayer = true;
	    histo->SetBinContent(nBinsX+1,binY,-1.);
	  }  else if(referenceCellOccup > 10 &&
		     nCellsZeroCount > nWires/3. &&
		     (double)nCellsZeroCount/(double)nWires >
		     2.*TMath::Erfc(referenceCellOccup/sqrt(referenceCellOccup))) {
	    LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << " -> would fail cells!" << endl;
	    LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "  # of cells with 0 count: " << nCellsZeroCount
							      << " # wires: " << nWires
							      << "  erfc: "
							      <<   TMath::Erfc(referenceCellOccup/sqrt(referenceCellOccup))
							      << endl;
	  }
      }
    }
    // Check if the whole layer is off
    if( nFailingLayers == 4) {
      nFailingSLs++;
      failSL = true;
    }
  }

  // All the chamber is off
  if(nFailingSLs == nSL) {
    chamberPercentage = 0;
    return 4;
  } else {
    chamberPercentage = 1.-(float)nFailingSLs/(float)nSL;
  }

  // FIXME add check on cells
  if(failSL) return 3;
  if(failLayer) return 2;
  if(failCells) return 1;

  return 0;
}


string DTOccupancyTest::topFolder() const {
  if(tpMode) return string("DT/10-TestPulses/");
  return string("DT/01-Digi/");
}

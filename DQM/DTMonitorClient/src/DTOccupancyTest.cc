

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/05/27 15:21:38 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - University and INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTOccupancyTest.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



using namespace edm;
using namespace std;




DTOccupancyTest::DTOccupancyTest(const edm::ParameterSet& ps){
  LogVerbatim ("DTOccupancyTest") << "[DTOccupancyTest]: Constructor";

  // Get the DQM service
  dbe = Service<DQMStore>().operator->();

}




DTOccupancyTest::~DTOccupancyTest(){
  LogVerbatim ("DTOccupancyTest") << " destructor called" << endl;


}




void DTOccupancyTest::beginJob(const EventSetup& context){
  LogVerbatim ("DTOccupancyTest") << "[DTOccupancyTest]: BeginJob";

  // Event counter
  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // Book the histos
  for(int wh = -2; wh <= 2; ++wh) { // loop over wheels
    bookHistos(wh, string("Occupancies"), "OccupancySummary");
  }

  dbe->setCurrentFolder("DT/Digi/");
  summaryHisto = dbe->book2D("OccupancySummary","Occupancy Summary",12,1,13,5,-2,3);
  summaryHisto->setAxisTitle("sector",1);
  summaryHisto->setAxisTitle("wheel",2);

}




void DTOccupancyTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  LogVerbatim ("DTOccupancyTest") <<"[DTOccupancyTest]: Begin of LS transition";
}




void DTOccupancyTest::analyze(const Event& e, const EventSetup& context) {
  nevents++;
  LogVerbatim ("DTOccupancyTest") << "[DTOccupancyTest]: "<<nevents<<" events";


}




void DTOccupancyTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  LogVerbatim ("DTOccupancyTest")
    <<"[DTOccupancyTest]: End of LS transition, performing the DQM client operation";
  
  // Reset the global summary
//   summaryHisto->Reset();


  // Get all the DT chambers
  vector<DTChamber*> chambers = muonGeom->chambers();

  for(vector<DTChamber*>::const_iterator chamber = chambers.begin();
      chamber != chambers.end(); ++chamber) {  // Loop over all chambers
    DTChamberId chId = (*chamber)->id();

    MonitorElement * chamberOccupancyHisto = dbe->get(getMEName("OccupancyAllHits_perCh", chId));	

    // Run the tests on the plot for the various granularities
    if(chamberOccupancyHisto != 0) {
      // Get the 2D histo
      TH2F* histo = chamberOccupancyHisto->getTH2F();
      int result = runOccupancyTest(histo, chId);
      int sector = chId.sector();
      if(sector == 13) { // FIXME: overwriting the previous value
	sector = 4;
      } else if(sector == 14) { // FIXME: overwriting the previous value
	sector = 10;
      }
      wheelHistos[chId.wheel()]->setBinContent(sector, chId.station(),result);
      if(result > summaryHisto->getBinContent(sector, chId.wheel())) {
	summaryHisto->setBinContent(sector, chId.wheel()+3, result);
      }
    } else {
      LogVerbatim ("DTOccupancyTest") << "[DTOccupancyTest] ME: "
				      << getMEName("OccupancyAllHits_perCh", chId) << " not found!" << endl;
    }

  }

  // Fill the global summary
  // Check for entire sectors off and report them on the global summary
  //FIXME: TODO



}


void DTOccupancyTest::endJob(){

  LogVerbatim ("DTOccupancyTest") << "[DTOccupancyTest] endjob called!";

//   if(dbe != 0) dbe->rmdir("DT/Digi");

}


  
// --------------------------------------------------
void DTOccupancyTest::bookHistos(const int wheelId, string folder, string histoTag) {
  // Set the current folder
  stringstream wheel; wheel << wheelId;	
  dbe->setCurrentFolder("DT/Digi/");

  // build the histo name
  string histoName = histoTag + "_W" + wheel.str(); 
  
  
  LogVerbatim ("DTOccupancyTest") <<"[DTOccupancyTest]: booking wheel histo:"<< endl
				  <<"              folder "<< "DT/Digi/Wheel"
    + wheel.str() + "/" + folder << endl
				  <<"              histoTag "<<histoTag << endl
				  <<"              histoName "<<histoName<<endl;
  
  string histoTitle = "Occupancy summary WHEEL: "+wheel.str();
  wheelHistos[wheelId] = dbe->book2D(histoName,histoTitle,12,1,13,4,1,5);
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


  string folderRoot = "DT/Digi/Wheel" + wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/";

  string folder = "Occupancies/";
  
  // build the histo name
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str();

  string histoname = folderRoot + folder + histoName;

  return histoname;
}




// Run a test on the occupancy of the chamber
// Return values:
// 0 -> all ok
// 1 -> # consecutive dead channels > N
// 2 -> dead layer
// 3 -> dead SL
// 4 -> dead chamber
int DTOccupancyTest::runOccupancyTest(const TH2F *histo, const DTChamberId& chId) const {
  int nBinsX = histo->GetNbinsX();

  bool failSL = false;
  bool failLayer = false;

  // Check that the chamber has digis
  if(histo->Integral() == 0) {
    return 4;
  }


  // Check the layer occupancy
  for(int slay = 1; slay <= 3; ++slay) { // loop over SLs
    // Skip layer 2 on MB4
    if(chId.station() == 4 && slay == 2) continue;
    // check the SL occupancy
    int binYlow = ((slay-1)*4)+1;
    int binYhigh = binYlow+3;
    if(histo->Integral(1,nBinsX,binYlow,binYhigh) == 0) {
      failSL = true;
    }
    for(int lay = 1; lay <= 4; ++lay) { // loop over layers
      int binY = binYlow+(lay-1);
//       DTLayerId layId(chId, slay, lay);
//       int nWires = muonGeom->layer((*ly)->id())->specificTopology().channels();
//       int firstWire = muonGeom->layer((*ly)->id())->specificTopology().firstChannel();
      if(histo->Integral(1,nBinsX,binY,binY) == 0) {
	failLayer = true;
      }
    }
  }

  // FIXME add check on cells
  if(failSL) return 3;
  if(failLayer) return 2;

  return 0;
}


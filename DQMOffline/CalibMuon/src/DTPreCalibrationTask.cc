#include "DQMOffline/CalibMuon/interface/DTPreCalibrationTask.h"


// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// Digis
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"


#include "TH2F.h"
#include "TFile.h"

using namespace edm;
using namespace std;



DTPreCalibrationTask::DTPreCalibrationTask(const edm::ParameterSet& ps){

  LogTrace("DTPreCalibSummary") <<"[DTPrecalibrationTask]: Constructor"<<endl;

  // Label to retrieve DT digis from the event
  digiLabel = consumes<DTDigiCollection>(ps.getUntrackedParameter<string>("digiLabel")); 

  // parameter for Time Boxes booking
  minTriggerWidth = ps.getUntrackedParameter<int>("minTriggerWidth",2000); 
  maxTriggerWidth = ps.getUntrackedParameter<int>("maxTriggerWidth",6000); 

  // get the histo folder name
  folderName = ps.getUntrackedParameter<string>("folderName");

}


DTPreCalibrationTask::~DTPreCalibrationTask(){}


void DTPreCalibrationTask::bookHistograms(DQMStore::IBooker &iBooker,
  edm::Run const &, edm::EventSetup const &) {

  for(int wheel=-2; wheel<=2; wheel++){
   for(int sector=1; sector<=14; sector++){
     LogTrace("DTPreCalibSummary") <<"[DTPrecalibrationTask]: Book histos for wheel "<<wheel<<", sector "<<sector<<endl;
     iBooker.setCurrentFolder(folderName+"/TimeBoxes");  
     bookTimeBoxes(iBooker, wheel, sector);
     iBooker.setCurrentFolder(folderName+"/OccupancyHistos");
     if(sector<13) bookOccupancyPlot(iBooker, wheel, sector);
    }
  }

}


void DTPreCalibrationTask::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  
  // Get the digis from the event
  edm::Handle<DTDigiCollection> dtdigis;
  event.getByToken(digiLabel, dtdigis);

  // LOOP OVER ALL THE DIGIS OF THE EVENT
  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){

      //Fill the Time Boxes
      int tdcTime = (*digiIt).countsTDC();
      TimeBoxes[make_pair((*dtLayerId_It).first.superlayerId().chamberId().wheel(),
			  (*dtLayerId_It).first.superlayerId().chamberId().sector())]->Fill(tdcTime);

      //Fill the occupancy plot
      const  DTLayerId dtLId = (*dtLayerId_It).first;
      int yBin = (dtLId.station()-1)*12+dtLId.layer()+4*(dtLId.superlayer()-1);
      if(dtLId.station()==4 && dtLId.superlayer()==3)
	yBin = (dtLId.station()-1)*12+dtLId.layer()+4*(dtLId.superlayer()-2);
      if((*dtLayerId_It).first.superlayerId().chamberId().sector()<13)
	OccupancyHistos[make_pair((*dtLayerId_It).first.superlayerId().chamberId().wheel(),
				  (*dtLayerId_It).first.superlayerId().chamberId().sector())]->Fill((*digiIt).wire(),yBin);
      else{
	if(dtLId.superlayer()!=3) yBin = 44 + dtLId.layer();
	else yBin = 48 + dtLId.layer();
	if((*dtLayerId_It).first.superlayerId().chamberId().sector()==13)
	  OccupancyHistos[make_pair((*dtLayerId_It).first.superlayerId().chamberId().wheel(),4)]->Fill((*digiIt).wire(),yBin);
	if((*dtLayerId_It).first.superlayerId().chamberId().sector()==14)
	  OccupancyHistos[make_pair((*dtLayerId_It).first.superlayerId().chamberId().wheel(),10)]->Fill((*digiIt).wire(),yBin);
      }
	
    }
  }

}

void DTPreCalibrationTask::bookTimeBoxes(DQMStore::IBooker &iBooker, int wheel, int sector) {

  stringstream wh; wh << wheel;
  stringstream sec; sec << sector;

  // book the time boxes
  TimeBoxes[make_pair(wheel, sector)]= iBooker.book1D("TimeBox_W"+wh.str()+"_Sec"+sec.str(), "Time Box W"+wh.str()+"_Sec"+sec.str(),(maxTriggerWidth-minTriggerWidth)/50, minTriggerWidth, maxTriggerWidth);
  TimeBoxes[make_pair(wheel, sector)]->setAxisTitle("TDC counts");

}



void DTPreCalibrationTask::bookOccupancyPlot(DQMStore::IBooker &iBooker, int wheel, int sector) {

  stringstream wh; wh << wheel;
  stringstream sec; sec << sector;

  // book the occpancy plot
  if(sector==4 || sector==10)
    OccupancyHistos[make_pair(wheel, sector)]= iBooker.book2D("Occupancy_W"+wh.str()+"_Sec"+sec.str(), "Occupancy W"+wh.str()+"_Sec"+sec.str(),100,1,100,52,1,53);
  else
    OccupancyHistos[make_pair(wheel, sector)]= iBooker.book2D("Occupancy_W"+wh.str()+"_Sec"+sec.str(), "Occupancy W"+wh.str()+"_Sec"+sec.str(),100,1,100,44,1,45);
  OccupancyHistos[make_pair(wheel, sector)]->setAxisTitle("wire number", 1);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(1,"M1L1",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(2,"M1L2",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(3,"M1L3",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(4,"M1L4",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(5,"M1L5",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(6,"M1L6",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(7,"M1L7",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(8,"M1L8",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(9,"M1L9",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(10,"M1L10",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(11,"M1L11",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(12,"M1L12",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(13,"M2L1",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(14,"M2L2",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(15,"M2L3",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(16,"M2L4",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(17,"M2L5",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(18,"M2L6",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(19,"M2L7",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(20,"M2L8",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(21,"M2L9",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(22,"M2L10",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(23,"M2L11",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(24,"M2L12",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(25,"M3L1",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(26,"M3L2",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(27,"M3L3",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(28,"M3L4",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(29,"M3L5",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(30,"M3L6",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(31,"M3L7",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(32,"M3L8",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(33,"M3L9",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(34,"M3L10",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(35,"M3L11",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(36,"M3L12",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(37,"M4L1",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(38,"M4L2",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(39,"M4L3",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(40,"M4L4",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(41,"M4L5",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(42,"M4L6",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(43,"M4L7",2);
  OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(44,"M4L8",2);
  if(sector==4){
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(45,"M4Sec13L1",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(46,"M4Sec13L2",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(47,"M4Sec13L3",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(48,"M4Sec13L4",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(49,"M4Sec13L5",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(50,"M4Sec13L6",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(51,"M4Sec13L7",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(52,"M4Sec13L8",2);
  }
  if(sector==10){
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(45,"M4Sec14L1",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(46,"M4Sec14L2",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(47,"M4Sec14L3",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(48,"M4Sec14L4",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(49,"M4Sec14L5",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(50,"M4Sec14L6",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(51,"M4Sec14L7",2);
    OccupancyHistos[make_pair(wheel, sector)]->setBinLabel(52,"M4Sec14L8",2);
  }

}

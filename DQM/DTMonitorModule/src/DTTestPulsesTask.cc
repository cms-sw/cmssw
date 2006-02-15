/*
 * \file DTTestPulsesTask.cc
 * 
 * $Date: 2006/02/02 18:27:32 $
 * $Revision: 1.2 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTTestPulsesTask.h>

#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

#include <CondFormats/DTObjects/interface/DTT0.h>
#include <CondFormats/DataRecord/interface/DTT0Rcd.h>


DTTestPulsesTask::DTTestPulsesTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe,
				   const edm::EventSetup& context){

  logFile.open("DTTestPulsesTask.log");

  string histoName;
  stringstream superLayer; 
  stringstream layer; 

  int minWHEEL = ps.getUntrackedParameter<int>("WheelToStartWith",0);
  int maxWHEEL = ps.getUntrackedParameter<int>("WheelToEndWith",0);
  int minSTATION = ps.getUntrackedParameter<int>("StationToStartWith",0);
  int maxSTATION = ps.getUntrackedParameter<int>("StationToEndWith",0);
  int minSECTOR = ps.getUntrackedParameter<int>("SectorToStartWith",0);
  int maxSECTOR = ps.getUntrackedParameter<int>("SectorToEndWith",0);

  // I'm interested only in the region around t0s peak
  /* Options:
     1) Get the mean value from the DB for a given chamber (== same cables);
     2) Set the mean by hand
   */
  if (!ps.getUntrackedParameter<bool>("t0sMeanFromDB",true))

    t0sPeakRange = make_pair( ps.getUntrackedParameter<int>("t0sRangeLowerBound", 6000), 
			      ps.getUntrackedParameter<int>("t0sRangeUpperBound", 6000) );


  if ( dbe ) {

    for (int w = minWHEEL; w <= maxWHEEL; ++w) {
      stringstream wheel; wheel << w;

      for (int st = minSTATION; st <= maxSTATION; ++st) {
	stringstream station; station << st;

	for (int sec = minSECTOR; sec <= maxSECTOR; ++sec) {
	  stringstream sector; sector << sec;


	  // get the mean value for a given chamber from the DB 
	  if (ps.getUntrackedParameter<bool>("t0sMeanFromDB",true)) {
	    
// 	    ESHandle<DTT0> t0Map;
// 	    context.get<DTT0Rcd>().get(t0Map);
//          t0Map.cellT0(w,st,sec,t0sMeanValueFromTP,t0RMSFromTP)

	    int t0sMeanValueFromTP = 6000;
	    int t0sRangeOfValidity = ps.getUntrackedParameter<int>("t0sRangeOfValidity",100);

	    t0sPeakRange = make_pair( t0sMeanValueFromTP - t0sRangeOfValidity/2 , 
				      t0sMeanValueFromTP + t0sRangeOfValidity/2 );
	  }

	  dbe->setCurrentFolder("DT/DTTestPulsesTask/Wheel" + wheel.str() +
				"/Station" + station.str() +
				"/Sector" + sector.str() + "TestPulses");

	  for (int sl = 1; sl <= 3; ++sl) {
	    superLayer << sl;
	    for (int l = 1; l <= 4; ++l) {
	      layer << l;
	      
	      // Here get the numbers of cells for layer from geometry
	      int number_of_cells = 100; //now set to the maximum

	      histoName = "TestPulse_SL" + superLayer.str() + "_L" + layer.str();
	      testPulsesHistos[int(DTLayerId(w,st,sec,sl,l).rawId())] =
                dbe->book2D(histoName,histoName,
			    number_of_cells, 0, number_of_cells, 
			    t0sPeakRange.first - t0sPeakRange.second, t0sPeakRange.first, t0sPeakRange.second);

	    }
	  }

	}
      }
    }



  }

}

DTTestPulsesTask::~DTTestPulsesTask(){

  logFile.close();

}

void DTTestPulsesTask::beginJob(const edm::EventSetup& c){

  nevents = 0;

}

void DTTestPulsesTask::endJob(){

  cout << "DTTestPulsesTask: analyzed " << nevents << " events" << endl;

}

void DTTestPulsesTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;
  
  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel("dtunpacker", dtdigis);
  
  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=dtdigis->begin(); detUnitIt!=dtdigis->end(); ++detUnitIt){
    
    for (DTDigiCollection::const_iterator digiIt = ((*detUnitIt).second).first;
	 digiIt!=((*detUnitIt).second).second; ++digiIt){
      
      // for clearness..
      int index = ((*detUnitIt).first).rawId();

      testPulsesHistos.find(index)->second->Fill(index,(*digiIt).countsTDC());
 

      
    }
  }

}


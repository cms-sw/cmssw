/*
 * \file DTTestPulsesTask.cc
 *
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTTestPulsesTask.h>

// Framework
#include <FWCore/Framework/interface/EventSetup.h>

// Digis
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// Pedestals
#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>
#include <CondFormats/DTObjects/interface/DTRangeT0.h>
#include <CondFormats/DataRecord/interface/DTRangeT0Rcd.h>
#include "DQMServices/Core/interface/DQMStore.h"


using namespace edm;
using namespace std;


DTTestPulsesTask::DTTestPulsesTask(const edm::ParameterSet& ps){


  cout<<"[DTTestPulseTask]: Constructor"<<endl;
  dtDigisToken_ = consumes<DTDigiCollection>(
      edm::InputTag(ps.getUntrackedParameter<std::string>("dtdigis", "dtunpacker")));

  parameters = ps;


  t0sPeakRange = make_pair( parameters.getUntrackedParameter<int>("t0sRangeLowerBound", -100),
			    parameters.getUntrackedParameter<int>("t0sRangeUpperBound", 100));

}

DTTestPulsesTask::~DTTestPulsesTask(){

  cout <<"[DTTestPulsesTask]: analyzed " << nevents << " events" << endl;

}

void DTTestPulsesTask::dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) {

   cout<<"[DTTestPulsesTask]: dqmBeginRun"<<endl;
  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);
  nevents = 0;
}

void DTTestPulsesTask::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & context) {

	  bookHistos( ibooker, string("TPOccupancy"), string("TestPulses") );
          bookHistos( ibooker, string("TPProfile"), string("TestPulses2D") );
          bookHistos( ibooker, string("TPTimeBox"), string("TestPulsesTB") );
}

void DTTestPulsesTask::bookHistos(DQMStore::IBooker & ibooker, string folder, string histoTag) {

  cout<<"[DTTestPulseTask]: booking"<<endl;

//here put the static booking loop

 // Loop over all the chambers
  vector<const DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  for (; ch_it != ch_end; ++ch_it) {

    // Loop over the SLs
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin();
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();

    for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	stringstream superLayer; superLayer << sl.superlayer();

	// Loop over the Ls
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin();
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();

	for(; l_it != l_end; ++l_it) {
	  DTLayerId layerId = (*l_it)->id();
	  stringstream layer; layer << layerId.layer();

          stringstream superLayer; superLayer << layerId.superlayer();
          stringstream station; station << layerId.superlayerId().chamberId().station();
          stringstream sector; sector << layerId.superlayerId().chamberId().sector();
          stringstream wheel; wheel << layerId.superlayerId().chamberId().wheel();

          // TP Profiles
          if ( folder == "TPProfile" ) {

	   const int nWires = (*l_it)->specificTopology().channels();

           ibooker.setCurrentFolder("DT/DTTestPulsesTask/Wheel" + wheel.str() +
		         	  "/Station" + station.str() +
			          "/Sector" + sector.str() +
			          "/SuperLayer" + superLayer.str() +
			           "/" +folder);

           string histoName = histoTag
                   + "_W" + wheel.str()
                   + "_St" + station.str()
                   + "_Sec" + sector.str()
                   + "_SL" + superLayer.str()
                   + "_L" + layer.str();

          // Setting the range
          if ( parameters.getUntrackedParameter<bool>("readDB", false) ) {
            t0RangeMap->slRangeT0( layerId.superlayerId() , t0sPeakRange.first, t0sPeakRange.second);
          }


          cout<<"t0sRangeLowerBound "<<t0sPeakRange.first<<"; "
	      <<"t0sRangeUpperBound "<<t0sPeakRange.second<<endl;


          testPulsesProfiles[int(DTLayerId(layerId.wheel(),
				   layerId.station(),
				   layerId.sector(),
				   layerId.superlayer(),
				   layerId.layer()).rawId())] =
          ibooker.bookProfile(histoName,histoName,
		                   nWires, 0, nWires, // Xaxis: channels
		                   t0sPeakRange.first - t0sPeakRange.second, t0sPeakRange.first, t0sPeakRange.second); // Yaxis: times
	  }

          // TP Occupancies
          else if ( folder == "TPOccupancy" ) {

          ibooker.setCurrentFolder("DT/DTTestPulsesTask/Wheel" + wheel.str() +
		        	  "/Station" + station.str() +
			          "/Sector" + sector.str() +
			          "/SuperLayer" + superLayer.str() +
			          "/" +folder);

           string histoName = histoTag
             + "_W" + wheel.str()
             + "_St" + station.str()
             + "_Sec" + sector.str()
             + "_SL" + superLayer.str()
             + "_L" + layer.str();

           const int nWires = muonGeom->layer(DTLayerId(layerId.wheel(),
	       					 layerId.station(),
	       					 layerId.sector(),
	       					 layerId.superlayer(),
						 layerId.layer()))->specificTopology().channels();

           testPulsesOccupancies[int(DTLayerId(layerId.wheel(),
					layerId.station(),
					layerId.sector(),
					layerId.superlayer(),
					layerId.layer()).rawId())] =
           ibooker.book1D(histoName, histoName, nWires, 0, nWires);
	  }

          // Time Box per Chamber
          else if ( folder == "TPTimeBox" ) {

          ibooker.setCurrentFolder("DT/DTTestPulsesTask/Wheel" + wheel.str() +
			          "/Station" + station.str() +
			          "/Sector" + sector.str() +
	         		  "/" +folder);

          string histoName = histoTag
                     + "_W" + wheel.str()
                     + "_St" + station.str()
                     + "_Sec" + sector.str();

          testPulsesTimeBoxes[int( DTLayerId(layerId.wheel(),
				       layerId.station(),
				       layerId.sector(),
				       layerId.superlayer(),
				       layerId.layer()).chamberId().rawId())] =
          ibooker.book1D(histoName, histoName, 10000, 0, 10000); // Overview of the TP (and noise) times
	  }

	} // close loop on layers
    } // close loop on superlayers
  } // close loop on chambers
}


void DTTestPulsesTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;

  edm::Handle<DTDigiCollection> dtdigis;
  e.getByToken(dtDigisToken_, dtdigis);

  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){

    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){

      // for clearness..
      int layerIndex = ((*dtLayerId_It).first).rawId();
      int chIndex = ((*dtLayerId_It).first).chamberId().rawId();


      if ((int)(*digiIt).countsTDC() > t0sPeakRange.first &&
	  (int)(*digiIt).countsTDC() < t0sPeakRange.second ) {

	// Occupancies

	  testPulsesOccupancies.find(layerIndex)->second->Fill((*digiIt).wire());

	// Profiles

	  testPulsesProfiles.find(layerIndex)->second->Fill((*digiIt).wire(),(*digiIt).countsTDC());
      }

        // Time Box

	testPulsesTimeBoxes.find(chIndex)->second->Fill((*digiIt).countsTDC());
    }
  }

}


// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:

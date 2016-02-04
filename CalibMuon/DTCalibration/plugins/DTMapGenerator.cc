/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/01/22 19:00:29 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara, S. Bolognesi - INFN Torino
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DTMapGenerator.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace edm;
using namespace std;

DTMapGenerator::DTMapGenerator(const ParameterSet& pset) {
  // The output file with the map
  outputMapName = pset.getUntrackedParameter<string>("outputFileName","output.map");
  // The input file with the base map (DDU ROS -> Wheel, Station, Sector)
  inputMapName = pset.getUntrackedParameter<string>("inputFileName","basemap.txt");
  //The ros type: ROS8 for commissioning, ROS25 otherwise
  rosType =  pset.getUntrackedParameter<int>("rosType",25);
  if(rosType != 8 && rosType != 25){
    cout<<"[DTMapGenerator]: wrong ros type (8 for commissioning, 25 otherwise)"<<endl;
    abort();
  }
}

DTMapGenerator::~DTMapGenerator(){}

void DTMapGenerator::endJob() {

  cout << "DTMapGenerator: Output Map: " << outputMapName << " ROS Type: " << rosType << endl;

  // Read the existing wires
  ifstream existingChannels("/afs/cern.ch/cms/Physics/muon/CMSSW/DT/channelsMaps/existing_channels.txt");
  
  set<DTWireId> wireMap; //FIXME:MAYBE YOU NEED THE > and == operators to use set?

  // Read the map between DDU - ROS and Chambers
  string lineMap;
  while (getline(existingChannels,lineMap)) {
    if( lineMap == "" || lineMap[0] == '#' ) continue; // Skip comments and empty lines
    stringstream linestr;
    linestr << lineMap;
    int wheelEx, stationEx, sectorEx, slEx, layerEx, wireEx;
    linestr >> wheelEx >> sectorEx >> stationEx >> slEx >> layerEx >> wireEx;
    DTWireId wireIdEx(wheelEx, stationEx, sectorEx, slEx, layerEx, wireEx);
    wireMap.insert(wireIdEx);
  }
  cout << "Map size: " << wireMap.size() << endl;

  // The map between DDU - ROS and Chambers
  ifstream skeletonMap(inputMapName.c_str());
  
  // The output map in the CMSSW format
  ofstream outputMap(outputMapName.c_str());

  // Read the map between DDU - ROS and Chambers
  string line;
  while (getline(skeletonMap,line)) {
    if( line == "" || line[0] == '#' ) continue; // Skip comments and empty lines
    stringstream linestr;
    linestr << line;
    int ddu, ros, wheel, station, sector;
    linestr >> ddu >> ros >> wheel >> station >> sector;
    cout << "DDU: " << ddu << endl
	 << "ROS: " << ros << endl
	 << "Connected to chamber in Wh: " << wheel << " St: " << station << " Sec: " << sector << endl;

    int previousROB = -1;
    int robCounter = -1;
    // The chamber map in ORCA commissioning format    
    string fileName;
    stringstream nameTmp;
    nameTmp << "/afs/cern.ch/cms/Physics/muon/CMSSW/DT/channelsMaps/templates/MB" << station << "_" <<  sector << ".map";
    nameTmp >> fileName;
    ifstream chamberMap(fileName.c_str());

    string lineChamberMap;
    while (getline(chamberMap,lineChamberMap)) {
      if( lineChamberMap == "" || lineChamberMap[0] == '#' ) continue; // Skip comments and empty lines
      stringstream chamberMapStr;
      chamberMapStr << lineChamberMap;
      
      int rob, tdc, tdcChannel, sl, layer, wire;
      int unusedRos, unusedChamberCode;
      int outRob = -1;
      chamberMapStr >> unusedRos  >> rob >> tdc >> tdcChannel >> unusedChamberCode >> sl >> layer >> wire;
      
      // Check if the channel really exists
      if(!checkWireExist(wireMap, wheel, station, sector, sl, layer, wire))
	continue;

      if(rob > previousROB) {
	previousROB = rob;
	robCounter++;
      } else if(rob < previousROB) {
	cout << "Error: ROB number is not uniformly increasing!" << endl;
	abort();
      }
      // Set the ROB id within the ros
      if(rosType == 25) {
	if(station == 1) {//MB1
	  outRob = robCounter;
	} else if(station == 2) {//MB2
	  outRob = robCounter + 6;
	} else if(station == 3) {//MB3
	  if(robCounter < 3) 
	    outRob = robCounter + 12;
	  else if(robCounter == 3)
	    outRob = 24;
	  else if(robCounter > 3)
	    outRob = robCounter + 11;
	} else if(station == 4) {//MB4
	  if(sector == 14)  {
	    if(robCounter == 3) {
	      continue;
	    }
	    outRob = robCounter + 18;
	  } else if(sector == 10) {
	    if(robCounter == 3) {
	      continue;
	    } else if(robCounter == 0) {
	      outRob = 21;
	    } else {
	      outRob = robCounter + 21;
	    }
	  }
	  if(sector == 4 )  {
	    if(robCounter == 3 || robCounter == 4 ) {
	      continue;
	    }
	    outRob = robCounter + 18;
	  } else if(sector == 13) {
	    if(robCounter == 3 || robCounter == 4) {
	      continue;
	    } else if(robCounter == 0) {
	      outRob = 21;
	    } else {
	      outRob = robCounter + 21;
	    }
	  } else if(sector == 11 || sector == 9) {
	    outRob = robCounter + 18;
	    if(robCounter == 3) {
	      continue;
	    }
	  }
	  //else if(sector==12 || sector == 8 || sector == 7 || sector == 6 || sector == 5 ||  sector == 3 || sector == 2 ||sector == 1 ){
	  else{
	    outRob = robCounter + 18;
	  }
	}
      } else {
	outRob = rob;
      }
      outputMap << ddu << " "
		<< ros << " "
		<< outRob << " "
		<< tdc << " "
		<< tdcChannel << " "
		<< wheel << " "
		<< station << " "
		<< sector << " "
		<< sl << " "
		<< layer << " "
		<< wire << endl;
    }
  }
}

bool DTMapGenerator::checkWireExist(const set<DTWireId>& wireMap, int wheel, int station, int sector, int sl, int layer, int wire) {
  DTWireId wireId(wheel, station, sector, sl, layer, wire);
  if(wireMap.find(wireId) == wireMap.end()) {
    cout << "Skipping channel: Wh: " << wheel
	 << ", st: " << station
	 << ", sec: " << sector
	 << ", sl: " << sl
	 << ", lay: " << layer
	 << ", wire: " << wire << endl;
    return false;
  }
  
  return true;
}



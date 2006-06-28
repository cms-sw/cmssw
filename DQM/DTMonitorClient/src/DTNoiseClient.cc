/*
 * \file DTNoiseClient.cc
 * 
 * $Date: 2006/06/27 16:37:51 $
 * $Revision: 1.7 $
 * \author M. Zanetti - INFN Padova
 *
 */

#include "DQM/DTMonitorClient/interface/DTNoiseClient.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

DTNoiseClient::DTNoiseClient() {
  
  //pippo  

}


DTNoiseClient::~DTNoiseClient() {
  
  ///~pippo

}


void DTNoiseClient::performCheck(MonitorUserInterface * mui) {

  /// FIXME: ES missing, no way to get the geometry 
  /// fake loop 
  for (int w=-2; w<=2; w++) {
    stringstream wheel; wheel  << w;
    for (int st=1; st<=4; st++) {
      stringstream station; station  << st;
      for (int se=1; se<=4; se++) {
	stringstream sector; sector  << se;

	/// WARNING: Pay attantion to the FU number!!!
	string folder = "Collector/FU0/DT/DTDigiTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/Occupancies/Noise/";

	for (int sl=1; sl<=3; sl++) {
	  if (se==4 && sl==2) continue;
	  stringstream superLayer; superLayer  << sl;
	  for (int l=1; l<=4; l++) {
	    stringstream layer; layer  << sl;
	    
	    string histoName = folder + 
	      + "OccupancyNoise_W" + wheel.str() 
	      + "_St" + station.str() 
	      + "_Sec" + sector.str() 
	      + "_SL" + superLayer.str() 
	      + "_L" + layer.str();
	    
	    MonitorElement * noise = mui->get(histoName);
	  }
	}
      }
    }
  }
}

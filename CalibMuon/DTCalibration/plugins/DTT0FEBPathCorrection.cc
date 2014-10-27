/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/04/10 17:55:08 $
 *  $Revision: 1.1 $
 *  \author Mark Olschewski
 */

#include "DTT0FEBPathCorrection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

#include <string>
#include <sstream>

using namespace std;
using namespace edm;

namespace dtCalibration {

DTT0FEBPathCorrection::DTT0FEBPathCorrection(const ParameterSet& pset):
   calibChamber_( pset.getParameter<string>("calibChamber") ) {

   //DTChamberId chosenChamberId;
   if( calibChamber_ != "" && calibChamber_ != "None" && calibChamber_ != "All" ){
      stringstream linestr;
      int selWheel, selStation, selSector;
      linestr << calibChamber_;
      linestr >> selWheel >> selStation >> selSector;
      chosenChamberId_ = DTChamberId(selWheel, selStation, selSector);
      LogVerbatim("Calibration") << "[DTT0FEBPathCorrection] Chosen chamber: " << chosenChamberId_ << endl;
   }
   //FIXME: Check if chosen chamber is valid.
}

DTT0FEBPathCorrection::~DTT0FEBPathCorrection() {
}

void DTT0FEBPathCorrection::setES(const EventSetup& setup) {
   // Get t0 record from DB
   ESHandle<DTT0> t0H;
   setup.get<DTT0Rcd>().get(t0H);
   t0Map_ = &*t0H;
   LogVerbatim("Calibration") << "[DTT0FEBPathCorrection] T0 version: " << t0H->version();
}

DTT0Data DTT0FEBPathCorrection::correction(const DTWireId& wireId) {
   // Compute for selected chamber (or All) correction using as reference chamber mean

   DTChamberId chamberId = wireId.layerId().superlayerId().chamberId();

   if( calibChamber_ == "" || calibChamber_ == "None" )          return defaultT0(wireId);
   if( calibChamber_ != "All" && chamberId != chosenChamberId_ ) return defaultT0(wireId);

   // Access DB
   float t0Mean,t0RMS;
   int status = t0Map_->get(wireId,t0Mean,t0RMS,DTTimeUnits::counts);
   if(status != 0) 
      throw cms::Exception("[DTT0FEBPathCorrection]") << "Could not find t0 entry in DB for" 
                                                                << wireId << endl;
   int wheel = chamberId.wheel();
   int station = chamberId.station();
   int sector = chamberId.sector();
   int sl = wireId.layerId().superlayerId().superlayer();
   int l = wireId.layerId().layer();
   int wire = wireId.wire();
   float t0MeanNew = t0Mean - t0FEBPathCorrection(wheel, station, sector, sl, l, wire);
   float t0RMSNew = t0RMS;
   return DTT0Data(t0MeanNew,t0RMSNew);
}

DTT0Data DTT0FEBPathCorrection::defaultT0(const DTWireId& wireId) {
   // Access default DB
   float t0Mean,t0RMS;
   int status = t0Map_->get(wireId,t0Mean,t0RMS,DTTimeUnits::counts);
   if(!status){
      return DTT0Data(t0Mean,t0RMS);
   } else{
      //... 
      throw cms::Exception("[DTT0FEBPathCorrection]") << "Could not find t0 entry in DB for"
	 << wireId << endl;
   }
}

/**

  Return the value to be subtracted to t0s to correct for the difference of 
  path lenght for TP signals within the FEB, from the TP input connector 
  to the MAD.

  FEBs are alternately connected on the right (J9) and left (J8) TP 
  input connectors.
  Path lenghts (provided from Franco Gonella) for FEB-16 channels 
  for FEB-16s connected on the right TP connector (J9) are:
 
  CH0,1 = 32 mm 
  CH2,3 = 42 mm 
  CH4,5 = 52 mm 
  CH6,7 = 62 mm 

  CH8,9 = 111 mm 
  CH10,11 = 121 mm 
  CH12,13 = 131 mm 
  CH14,15 = 141 mm

  Given that ttrig calibration absorbs average offsets, 
  we assume thate only differences w.r.t. the average lenght (86.5 mm) remain.

  For FEBs connected on the right connector, values are swapped; so there is 
  a periodicity of 2 FEBS (8 cells)

  The mapping of FEB channels to wires, for the first two FEBs, is: 

         FEB 0 (J9)     FEB 1 (J8)     
  Wire | 1  2  3  4  |  5  6  7  8
  --------------------------------  
   L1  | 3  7 11 15  |  3  7 11 15 
   L2  | 1  5  9 13  |  1  5  9 13
   L3  | 2  6 10 14  |  2  6 10 14
   L4  | 0  4  8 12  |  0  4  8 12


  For FEB-20, distances from the left connector (J8) are:
  CH16,17 = 171 mm  
  CH18,19 = 181 mm

  We do not include a correction for this additional row of channels since
  they are at the edge of the SL so the effect cannot be seen on data (and moreover 
  the channel in L1 is usually not existing in the chamber)

*/

float DTT0FEBPathCorrection::t0FEBPathCorrection(int wheel, int st, int sec, int sl, int l, int w) {

  // Skip correction for the last row of cells of FEB20 (see above)
  if( (st==1 && ((sl!=2 && w ==49) || (sl==2 && w ==57))) || 
      ((st==2||st==3)&& (sl==2 && w ==57)) ) return 0.;

  
  float dist[8]={};

  // Path lenght differences for L1 and L3 (cm)
  if(l==1 || l ==3){
   
    dist[0] = +4.45;
    dist[1] = +2.45;
    dist[2] = -3.45;
    dist[3] = -5.45;
    dist[4] = -4.45;
    dist[5] = -2.45;
    dist[6] = +3.45;
    dist[7] = +5.45;
  }

  // Path lenght differences for L2 and L4 (cm)
  else {
    
    dist[0] = +5.45;
    dist[1] = +3.45;
    dist[2] = -2.45;
    dist[3] = -4.45;
    dist[4] = -5.45;
    dist[5] = -3.45;
    dist[6] = +2.45;
    dist[7] = +4.45;
  }
  
  
  // Wire position within the 8-cell period (2 FEBs). 
  // Note that wire numbers start from 1.
  int pos = (w-1)%8;

  // Special case: in MB2 phi and MB4-10, the periodicity is broken at cell 49, as there is 
  // an odd number of FEDs (15): the 13th FEB is connected on the left, 
  // and not on the right; i.e. one FEB (4 colums of cells) is skipped from what would 
  // be the regular structure.
  // The same happens in MB4-8/12 at cell 81.
  if ((st==2 && sl!=2 && w>=49) || 
      (st==4 && sec==10 && w>=49) || 
      (st==4 && (sec==8||sec==12) && w>=81) ) pos =(w-1+4)%8;

  // Inverse of the signal propagation speed, determined from the 
  // observed amplitude of the modulation. This matches what is found 
  // with CAD simulation using reasonable assumptions on the PCB specs.

  return dist[pos]*0.075;
  
}





} // namespace

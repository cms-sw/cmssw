#include "L1Trigger/DTTriggerPhase2/interface/MPThetaMatching.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

using namespace edm;
using namespace std;
using namespace cmsdt;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPThetaMatching::MPThetaMatching(const ParameterSet &pset)
  : MPFilter(pset), debug_(pset.getUntrackedParameter<bool>("debug")), 
                    th_option_(pset.getParameter<int>("th_option")),
                    th_quality_(pset.getParameter<int>("th_quality")),
                    scenario_(pset.getParameter<int>("scenario")) {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MPThetaMatching::initialise(const edm::EventSetup &iEventSetup) {}

void MPThetaMatching::run(edm::Event &iEvent,
                      const edm::EventSetup &iEventSetup,
                      std::vector<metaPrimitive> &inMPaths,
		      std::vector<metaPrimitive> &outMPaths) {

  if (debug_)
    LogDebug("MPThetaMatching") << "MPThetaMatching: run";
 
  double shift_back = 0; // Needed for t0 (TDC) calculation, taken from main algo
  if (scenario_ == MC) shift_back = 400;
  else if (scenario_ == DATA) shift_back = 0;
  else if (scenario_ == SLICE_TEST) shift_back = 400;

  auto filteredMPs = filter(inMPaths, th_option_, th_quality_, shift_back);
  for (auto &mp : filteredMPs)
    outMPaths.push_back(mp);
  
}

void MPThetaMatching::finish(){};

///////////////////////////
///  OTHER METHODS

std::vector<metaPrimitive> MPThetaMatching::filter(std::vector<metaPrimitive> inMPs,  
                                                       int th_option, int th_quality, double shift_back) {
  std::vector<metaPrimitive> outMPs;
  std::vector<metaPrimitive> thetaMPs;
  std::vector<metaPrimitive> phiMPs;

  //survey theta and phi MPs
  for (auto & mp: inMPs) {

   DTChamberId chId(mp.rawId);
   DTSuperLayerId slId(mp.rawId);

   if(slId.superLayer() == 2) thetaMPs.push_back(mp); 
   else phiMPs.push_back(mp);
  }

  // Loop on phi, save those at station without Theta MPs
  for (auto & mp: phiMPs) {

   DTChamberId chId(mp.rawId);
   DTSuperLayerId slId(mp.rawId);

   int sector  = chId.sector();
   int wheel   = chId.wheel();
   int station = chId.station();

   if (station == 4) {outMPs.push_back(mp); //No theta matching for MB4, save MP
                                       continue;}

   if(!isThereThetaMPInChamber(sector,wheel,station,thetaMPs) ) {outMPs.push_back(mp); // No theta MPs in chamber to match, save MP
                           continue;}
   }
 
   
  // Loop on theta
  for (auto & mp1: thetaMPs) {
   std::vector<std::pair<metaPrimitive,float>> deltaTimePosPhiCands;

   DTChamberId chId(mp1.rawId);
   DTSuperLayerId slId(mp1.rawId);

   int sector  = chId.sector();
   int wheel   = chId.wheel();
   int station = chId.station();
  
   if (station == 4) { LogDebug("MPThetaMatching") << "MPThetaMatching: station 4 does NOT have Theta SL 2"; 
                       continue;}

//    float t0 = (mp1.t0 - shift_back * LHC_CLK_FREQ) * ((float) TIME_TO_TDC_COUNTS / (float) LHC_CLK_FREQ); 
    float t0 = ((int)round(mp1.t0 / (float)LHC_CLK_FREQ)) - shift_back;
    float posRefZ = zFE[wheel+2];
    if(wheel==0 && (sector ==1 || sector==4 || sector==5 || sector==8 || sector==9 ||sector==12)) posRefZ = -posRefZ;
    float posZ = abs(mp1.phi); //???
    
    for (auto & mp2: phiMPs) {

     DTChamberId chId2(mp2.rawId);
     DTSuperLayerId slId2(mp2.rawId);

     //if      (co_option==1 && PhiMP2==0) continue; // Phi Only
     //else if (co_option==2 && PhiMP2==1) continue; // Theta Only

     int sector2  = chId2.sector();
     int wheel2   = chId2.wheel();
     int station2 = chId2.station();
     if (station2 == 4) continue; 
    
     if(station2 != station || sector2 != sector || wheel2 != wheel) continue; 
     
     if ((mp2.quality > th_quality)) {outMPs.push_back(mp2); //don't do theta matching for q > X, save
                                       continue; 
     }

//     float t02 = (mp2.t0 - shift_back * LHC_CLK_FREQ) * ((float) TIME_TO_TDC_COUNTS / (float) LHC_CLK_FREQ);
     float t02 = ((int)round(mp2.t0 / (float)LHC_CLK_FREQ)) - shift_back;

     //cout<<"posRefZ: "<<posRefZ<<" Z: "<<posZ/ZRES_CONV<<endl;
     float tphi = t02-abs(posZ/ZRES_CONV -posRefZ)/vwire;
     //cout<<"tphi: "<<tphi<<endl;
     
     int LR = -1;
     if(wheel==0 && (sector==3 || sector==4 || sector==7 || sector==8 || sector==11 || sector==12)) LR = +1;
     else if (wheel>0) LR = pow(-1,wheel+sector+1);
     else if (wheel<0) LR = pow(-1,-wheel+sector);
     //cout<<"wh st se: "<< wheel <<" "<< station <<" "<< sector <<" LR: "<< LR<<endl;
     float posRefX = LR*xFE[station-1];
     float ttheta =t0-(mp2.x *1000 -posRefX)/vwire;

     deltaTimePosPhiCands.push_back({mp2,abs(tphi-ttheta)});

  
    } //loop in phis

    //reorder deltaTimePosPhiCands according to tphi-ttheta distance
    std::sort(deltaTimePosPhiCands.begin(), deltaTimePosPhiCands.end(), compare);
    int count = 0;
    for (const std::pair<metaPrimitive, float>& p : deltaTimePosPhiCands){ //save up to nth nearest Phi candidate
       if(count<th_option)
       outMPs.push_back(p.first); 
       else break;
       count++;
       }
   }// loop in thetas

  return outMPs;
}

bool MPThetaMatching::isThereThetaMPInChamber(int sector2,int wheel2,int station2, std::vector<metaPrimitive> thetaMPs){
  for (auto & mp1: thetaMPs) {
   DTChamberId chId(mp1.rawId);
   DTSuperLayerId slId(mp1.rawId);

   int sector  = chId.sector();
   int wheel   = chId.wheel();
   int station = chId.station();
   if (sector==sector2 && wheel == wheel2 && station==station2) return true;
}
   return false;
};


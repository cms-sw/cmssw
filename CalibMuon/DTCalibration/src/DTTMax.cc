/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/05/23 15:28:45 $
 *  $Revision: 1.9 $
 *  \author Marina Giunta
 */

#include "CalibMuon/DTCalibration/interface/DTTMax.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
// Declare histograms for debugging purposes
#include "CalibMuon/DTCalibration/interface/Histogram.h"

#include <map>
#include <iostream>

using namespace std;
using namespace dttmaxenums;
using namespace DTEnums;

DTTMax::InfoLayer::InfoLayer(const DTRecHit1D& rh_, const DTSuperLayer & isl, GlobalVector dir,
			     GlobalPoint pos, DTTTrigBaseSync* sync):
  rh(rh_), idWire(rh.wireId()), lr(rh.lrSide()) {
    const DTLayer*  layer = isl.layer(idWire.layerId());
    LocalPoint wirePosInLayer(layer->specificTopology().wirePosition(idWire.wire()), 0, 0);
    LocalPoint wirePosInSL=isl.toLocal(layer->toGlobal(wirePosInLayer));
    wireX = wirePosInSL.x();
    
    //-- Correction for signal propagation along wire, t0 subtraction,
    LocalVector segDir =  layer->toLocal(dir);
    LocalPoint segPos = layer->toLocal(pos);
    LocalPoint segPosAtLayer = segPos + segDir*(-segPos.z())/cos(segDir.theta());
    LocalPoint hitPos(rh.localPosition().x() ,segPosAtLayer.y(),0.);
    time = rh.digiTime() - sync->offset(layer, idWire, layer->toGlobal(hitPos));
 
    if (time < 0. || time > 415.) {
      // FIXME introduce time window to reject "out-of-time" digis
      cout << " *** WARNING time = " << time << endl;
    }
  }


DTTMax::DTTMax(const vector<DTRecHit1D>& hits, const DTSuperLayer & isl, GlobalVector dir, 
	       GlobalPoint pos, DTTTrigBaseSync* sync):
  theInfoLayers(4,(InfoLayer*)0), //FIXME
  theTMaxes(4,(TMax*)0) 
{
  // debug parameter for verbose output
  debug = "true";

  // Collect all information using InfoLayer
  for (vector<DTRecHit1D>::const_iterator hit=hits.begin(); hit!=hits.end();
       ++hit) {
    //     cout << "Hit Pos " << (*hit).localPosition() << endl;
    
    InfoLayer* layInfo = new InfoLayer((*hit), isl, dir, pos, sync);
    int ilay = layInfo->idWire.layer();
    if (getInfoLayer(ilay)==0) {
      getInfoLayer(ilay) = layInfo;
    } else {
      // FIXME: in case there is > 1 hit/layer, the first is taken and the others are IGNORED.
      delete layInfo;
    }
  }

  // Get the segment direction
  theSegDir = ((isl.toLocal(dir).x() < 0)? L : R);

  int layersIn = 0; 
  int nGoodHits=0;
  for(vector<InfoLayer*>::const_iterator ilay =  theInfoLayers.begin();
      ilay != theInfoLayers.end(); ilay++) {
    if ((*ilay) == 0 ) {
      theSegType+= "X";
      continue;
    }
    DTEnums::DTCellSide lOrR =(*ilay)->lr;
    if(lOrR == Left) theSegType+= "L";
    else if (lOrR == Right) theSegType+="R"; 
    else theSegType+= "X";
    
    // layersIn : 6 =  layers 1,2,3
    //            7 =         1,2,4
    //            8 =         1,3,4
    //            9 =         2,3,4
    //            10=         1,2,3,4  
    layersIn += (*ilay)->idWire.layer();
    nGoodHits++;
  }

  if(nGoodHits >=3 && (theSegType != "RRRR" && theSegType != "LLLL")) {
    float t1 = 0.;
    float t2 = 0.;
    float t3 = 0.;
    float t4 = 0.;
    float x1 = 0.;
    float x2 = 0.;
    float x3 = 0.;
    float x4 = 0.;

    if(layersIn <= 8  || layersIn == 10) {
      t1 = getInfoLayer(1)->time; 	
      x1 = getInfoLayer(1)->wireX;      
    }
    if(layersIn <= 7  || layersIn >= 9) {
      t2 = getInfoLayer(2)->time; 
      x2 = getInfoLayer(2)->wireX;	      
    }
    if(layersIn == 6  || layersIn >= 8) {
      t3 = getInfoLayer(3)->time; 
      x3 = getInfoLayer(3)->wireX;	      
    }
    if( layersIn >= 7) {
      t4 = getInfoLayer(4)->time;
      x4 = getInfoLayer(4)->wireX;
    }
    
    float t = 0.;
    TMaxCells cGroup = notInit;
    string type;
    SigmaFactor sigma = noR; // Return the factor relating the width of the TMax distribution and the cell resolution
    float halfCell = 2.1;    // 2.1 is the half cell length in cm
    float delta = 0.5;       // (diff. wire pos.) < delta, halfCell+delta, .....
    unsigned t0Factor = 99;  // "quantity" of Delta(t0) included in the tmax formula

    //Debug
    if (debug) {
      cout << "seg. type: " << theSegType << " and dir: " << theSegDir << endl;
      cout << "t1, t2, t3, t4: " << t1 << " " << t2 << " " << t3 << " " << t4 << endl;
      cout << "x1, x2, x3, x4: " << x1 << " " << x2 << " " << x3 << " " << x4 << endl;
    }
    
    //different t0 hists (if you have at least one hit within a certain distance from the wire)
    unsigned hSubGroup = 99; //
    if(t1 == 0. || t2 == 0. || t3 == 0. || t4 == 0.)
      hSubGroup = 0; //if only 3 hits
    else if(t1<=5. || t2<=5. || t3<=5. || t4<=5.)
      hSubGroup = 1; //if distance of one hit from wire < 275um (v_drift=55um/ns) 
    else if(t1<=10. || t2<=10. || t3<=10. || t4<=10.)
      hSubGroup = 2;
    else if(t1<=20. || t2<=20. || t3<=20. || t4<=20.)
      hSubGroup = 3;
    else if(t1<=50. || t2<=50. || t3<=50. || t4<=50.)
      hSubGroup = 4;

    if((layersIn == 6 || layersIn == 10) && (fabs(x1-x3)<delta)) {
      cGroup = c123;
      ((type+=theSegType[0])+=theSegType[1])+=theSegType[2];
      sigma = r32;
      if(type == "LRL" || type == "RLR") {
	t0Factor = 2;
	t = (t1+t3)/2.+t2;
	hT123LRL->Fill(t);
      }
      else if((type == "LLR" && theSegDir == R) ||
	      (type == "RRL" && theSegDir == L)) {
	t0Factor = 1;
	t = (t3-t1)/2.+t2;
	hT123LLR->Fill(t);
      }
      else if((type == "LRR" && theSegDir == R) ||
	      (type == "RLL" && theSegDir == L)) {
	t0Factor = 1;
	t = (t1-t3)/2.+t2;
	hT123LRR->Fill(t);
      }
      else {
	t = -1.;
	sigma = noR;
	hT123Bad->Fill(t);
      }
      theTMaxes[cGroup] = new TMax(t,cGroup,type,sigma,t0Factor,hSubGroup);
      if(debug) cout << "tmax123 " << t << " " << type << endl;
    }
    if(layersIn == 7 || layersIn == 10) {
      cGroup = c124;
      type.clear();
      sigma = r72;
      ((type+=theSegType[0])+=theSegType[1])+=theSegType[3];
      if((theSegType == "LRLR" && type == "LRR" && x1 > x4) ||
	 (theSegType == "RLRL" && type == "RLL" && x1 < x4)) {
	t0Factor = 2;
	t = 1.5*t2+t1-t4/2.;
	hT124LRR1gt4->Fill(t);
      }
      else if((type == "LLR" && theSegDir == R && (fabs(x2-x4)<delta) && x1 < x2) ||
	      (type == "RRL" && theSegDir == L && (fabs(x2-x4)<delta) && x1 > x2)) {
	t0Factor = 1;
	t = 1.5*t2-t1+t4/2.;
	hT124LLR->Fill(t);
      }
      else if((type == "LLL" && theSegDir == R && (fabs(x2-x4)<delta) && x1 < x2) ||
	      (type == "RRR" && theSegDir == L && (fabs(x2-x4)<delta) && x1 > x2)) {
	t0Factor = 0;
	t = 1.5*t2-t1-t4/2.;
	hT124LLLR->Fill(t);
      }
      else if((type == "LLL" && theSegDir == L && (fabs(x2-x4)<delta)) ||
	      (type == "RRR" && theSegDir == R && (fabs(x2-x4)<delta))) {
	t0Factor = 0;
	t = -1.5*t2+t1+t4/2.;
	hT124LLLL->Fill(t);
      } 
      else if((type == "LRL" && theSegDir == L && (fabs(x2-x4)<delta)) ||
	      (type == "RLR" && theSegDir == R && (fabs(x2-x4)<delta))) {
	t0Factor = 3;
	t = 1.5*t2+t1+t4/2.;
	hT124LRLL->Fill(t);
      }
      else if((type == "LRL" && theSegDir == R && (fabs(x1-x4)<(halfCell+delta))) ||
	      (type == "RLR" && theSegDir == L && (fabs(x1-x4)<(halfCell+delta)))) {
	t0Factor = 99;  // it's actually 1.5, but this value it's not used  
	t = 3./4.*t2+t1/2.+t4/4.;
	sigma = r78;
	hT124LRLR->Fill(t);
      }
      else if((type == "LRR" && theSegDir == R && x1 < x4 && (fabs(x1-x4)<(halfCell+delta)))||
	       (type == "RLL" && theSegDir == L && x1 > x4 && (fabs(x1-x4)<(halfCell+delta)))) {
	t0Factor = 1;
	t = 3./4.*t2+t1/2.-t4/4.;
	sigma = r78;
	hT124LRR1lt4->Fill(t);
      }
      else {
	t = -1.; 
	sigma = noR;
	hT124Bad->Fill(t);
      }
      theTMaxes[cGroup] = new TMax(t,cGroup,type,sigma,t0Factor,hSubGroup);
      if(debug) cout << "tmax124 " << t << " " << t0Factor << " "  << type << endl;
    }
    if(layersIn == 8 || layersIn == 10) {
      cGroup = c134;
      type.clear();
      ((type+=theSegType[0])+=theSegType[2])+=theSegType[3];
      sigma = r72;
      if((type == "LLR" && x1 > x4 && theSegType == "LRLR") ||
	 (type == "RRL" && x1 < x4 && theSegType == "RLRL")) {
	t0Factor = 2;
	t = 1.5*t3+t4-t1/2.;
	hT134LLR1gt4->Fill(t);
      }
      else if((type == "LLR"  && x1 < x4 && (fabs(x1-x4)<(halfCell+delta))) ||
	       (type == "RRL"  && x1 > x4 && (fabs(x1-x4)<(halfCell+delta)))) {
	t0Factor = 1;
	t = 3./4.*t3+t4/2.-t1/4.; 
	sigma = r78;
	hT134LLR1lt4->Fill(t);
      }
      else if((type == "LRR"  && theSegDir == R && x1 < x4 && (fabs(x1-x3)<delta)) ||
	       (type == "RLL"  && theSegDir == L && x1 > x4 &&(fabs(x1-x3)<delta))) {
	t0Factor = 1;
	t = 1.5*t3-t4+t1/2.;
	hT134LRR->Fill(t);
      }
      else if((type == "LRL"  && theSegDir == R && (fabs(x1-x3)<delta)) ||
	      (type == "RLR"  && theSegDir == L && (fabs(x1-x3)<delta))) {
	t0Factor = 3;
	t = 1.5*t3+t4+t1/2.;
	hT134LRLR->Fill(t);
      }
      else if((type == "LRL"  && theSegDir == L && (fabs(x1-x3)<(2.*halfCell+delta))) ||
	      (type == "RLR"  && theSegDir == R && (fabs(x1-x3)<(2.*halfCell+delta)))) {
	t0Factor = 99; // it's actually 1.5, but this value it's not used  
	t = 3./4.*t3+t4/2.+t1/4.;
	sigma = r78;
	hT134LRLL->Fill(t);
      }
      else if((type == "LLL"  && theSegDir == L && x1 > x4 && (fabs(x1-x3)<delta)) ||
	      (type == "RRR"  && theSegDir == R && x1 < x4 && (fabs(x1-x3)<delta))) {
	t0Factor = 0;
	t = 1.5*t3-t4-t1/2.;
	hT134LLLL->Fill(t);	
      }
      else if((type == "LLL"  && theSegDir == R && (fabs(x1-x3)<delta)) ||
	      (type == "RRR"  && theSegDir == L && (fabs(x1-x3)<delta))) {
	t0Factor = 0;
	t = -1.5*t3+t4+t1/2.;
	hT134LLLR->Fill(t);
      }
      else {
	t = -1;
	sigma = noR;
	hT134Bad->Fill(t);
      }
      theTMaxes[cGroup] = new TMax(t,cGroup,type,sigma,t0Factor,hSubGroup);
      if(debug) cout << "tmax134 " << t << " " << t0Factor << " "  << type << endl;
    }
    if((layersIn == 9 || layersIn == 10) && (fabs(x2-x4)<delta)) {
      cGroup = c234;
      type.clear();
      ((type+=theSegType[1])+=theSegType[2])+=theSegType[3];
      sigma = r32;
      if((type == "LRL" ) ||
	 (type == "RLR" )) {
	t0Factor = 2;
	t = (t2+t4)/2.+t3;
	hT234LRL->Fill(t);
      }
      else if((type == "LRR" && theSegDir == R) ||
	      (type == "RLL" && theSegDir == L)) {
	t0Factor = 1;
	t = (t2-t4)/2.+t3;
	hT234LRR->Fill(t);
      }
      else if((type == "LLR" && theSegDir == R) ||
	      (type == "RRL" && theSegDir == L)) {
	t0Factor = 1;
	t = (t4-t2)/2.+t3;
	hT234LLR->Fill(t);
      }
      else {
	t = -1;
	sigma = noR;
	hT234Bad->Fill(t);
      }
      theTMaxes[cGroup] = new TMax(t,cGroup,type,sigma,t0Factor,hSubGroup);
      if(debug) cout << "tmax234 " << t << " " << type << endl;
    }
  }      
  
}


vector<const DTTMax::TMax*> DTTMax::getTMax(const DTWireId & idWire) {
  vector<const TMax*> v;
  if(idWire.layer()==1) {
    v.push_back(getTMax(c123)); //FIXME: needs pointer
    v.push_back(getTMax(c124));
    v.push_back(getTMax(c134));
  }
  else if(idWire.layer()==2) {
    v.push_back(getTMax(c123));
    v.push_back(getTMax(c124));
    v.push_back(getTMax(c234));
  }
  else if(idWire.layer()==3) {
    v.push_back(getTMax(c123));
    v.push_back(getTMax(c134));
    v.push_back(getTMax(c234));
  }
  else {
    v.push_back(getTMax(c124));
    v.push_back(getTMax(c134));
    v.push_back(getTMax(c234));
  }
  return v;
}



vector<const DTTMax::TMax*> DTTMax::getTMax(const DTSuperLayerId & isl) {
  vector<const TMax*> v;
  // add TMax* to the vector only if it really exists 
  if(getTMax(c123)) v.push_back(getTMax(c123)); 
  if(getTMax(c124)) v.push_back(getTMax(c124));
  if(getTMax(c134)) v.push_back(getTMax(c134));
  if(getTMax(c234)) v.push_back(getTMax(c234));  
  return v;
}


const DTTMax::TMax* DTTMax::getTMax(TMaxCells cCase){
  return theTMaxes[cCase];
}

/* Destructor */ 
DTTMax::~DTTMax(){
  for (vector<InfoLayer*>::const_iterator ilay = theInfoLayers.begin();
       ilay != theInfoLayers.end(); ilay++) {
    delete (*ilay);
  }

  for (vector<TMax*>::const_iterator iTmax = theTMaxes.begin();
       iTmax != theTMaxes.end(); iTmax++) {
    delete (*iTmax);
  }
}

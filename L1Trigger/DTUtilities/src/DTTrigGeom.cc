//-------------------------------------------------
//
//   Class: DTTrigGeom
//
//   Description: Muon Barrel Trigger Geometry
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   S. Vanini : NEWGEO implementation
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"

//-------------
// C Headers --
//-------------

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <iomanip>

using namespace std;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"

//----------------
// Constructors --
//----------------

DTTrigGeom::DTTrigGeom(DTChamber* stat, DTConfig* conf) : 
  _stat(stat), _config(conf) {

  // get the geometry from the station
  getGeom();

}


//--------------
// Destructor --
//--------------

DTTrigGeom::~DTTrigGeom() {}


//--------------
// Operations --
//--------------

float 
DTTrigGeom::phiSLOffset(){
  //sl1 offset respect to sl3 - in Front End view!!
  float x1 = tubePosInCh(1,1,1).x();
  float x3 = tubePosInCh(3,1,1).x();
  float offset = x1-x3;
  //   if(posFE(1)==1)        // Obsolete in
  //     offset = - offset;   // CMSSW
  
  return offset;
}

/* OBSOLETE - MAYBE - 19/06/06
int
DTTrigGeom::layerFEStaggering(int nsl, int nlay) const {

  NB the staggering is in FE view!
  return cell staggering respect to first wire in layer 1 in cell units
  the following is the default: staggering 0 of each layer 
     +---------+---------+---------+
     | 1  o    | 2  o    | 3  o    |
     +----+----+----+----+----+----+
          | 1  o    |  2 o    |
     +----+----+----+----+----+
     | 1  o    | 2  o    |
     +----+----+----+----+----+
          | 1  o    | 2  o    |
          +---------+---------+   ------------> x (y coming out of video) in SL frame
 

  int stag = 0;

  if(station()==4 && nsl==2){
    std::cout << "No theta superlayer in station 4!" << std::endl;
    return 0;
  }
  //position in chamber of wire 1 in layer 1
  LocalPoint posInCh_lay1      = tubePosInCh(nsl,1,1);
  //position in chamber of wire 1 in layer nlay
  int n1stwire = (nlay==4 ? 2 : 1); 
  LocalPoint posInCh_lay       = tubePosInCh(nsl,nlay,n1stwire);

 cout << endl;
 cout << nlay << posInCh_lay1 << posInCh_lay << endl;
 cout <<endl ;


  //NB PITCH-0.01 for computer approximation bug
  if(nsl==2){//SL2: first wire is toward positive y 
    stag = static_cast<int>((-posInCh_lay.y()+posInCh_lay1.y())/(_PITCH-0.01) + 0.5*(fmod(nlay,2.)==0?1:0));
  }
  else{//SL1 and SL3: first wire is toward negative x
	  if (nlay==4) {
		  stag = static_cast<int>((+posInCh_lay.x()-posInCh_lay1.x()+_PITCH)/(_PITCH-0.01) + 0.5*(fmod(nlay,2.)==0?1:0));
	  }
	  else {
		stag = static_cast<int>((+posInCh_lay.x()-posInCh_lay1.x())/(_PITCH-0.01) + 0.5*(fmod(nlay,2.)==0?1:0));
	  }
  }

  //FEP=1 means in y negative in layer frame
  const DTLayer* lay  = _stat->superLayer(DTSuperLayerId(statId(),nsl))->layer(DTLayerId(statId(),nsl,nlay));
  const DTLayer* lay1 = _stat->superLayer(DTSuperLayerId(statId(),nsl))->layer(DTLayerId(statId(),nsl,1));   

  if(lay->getFEPosition()==1){ //FE staggering is reverted                                    //MODIFICARE DOPO!!!!!!
    int nWire  = lay->specificTopology().channels();  
    int nWire1 = lay1->specificTopology().channels();
    stag = - nWire + nWire1 - stag + (fmod(nlay,2.)==0?1:0);
  }
  return stag;
}
*/

int
DTTrigGeom::mapTubeInFEch(int nsl, int nlay, int ntube) const {
  int nch = 0;
  if(station()==4 && nsl==2){
    std::cout << "No theta superlayer in station 4!" << std::endl;
  }
  else{
    // obsolete 19/06/2006  const DTLayer* lay = _stat->superLayer(DTSuperLayerId(statId(),nsl))->layer(DTLayerId(statId(),nsl,nlay));
	  
/* obsolete 19/6/06
 if(lay->getFEPosition()==0)         //FE is in Y negative: opposite numbering                
	  nch = lay->specificTopology().channels() - ntube + 1;	  
//   if(lay->getFEPosition()==1)         //FE is in Y positive: same numbering digi-trig        
//     nch = ntube;
//  }
*/	
	// in new geometry depends on SL: theta tube numbering is reverted wrt hardware
	nch =ntube;
/*	if(nsl==2){	
		nch = lay->specificTopology().channels() - ntube + 1;
	}*/
  }
  return nch;
}

LocalPoint 
DTTrigGeom::tubePosInCh(int nsl, int nlay, int ntube) const {
  if ( nlay==4 && ntube==1) {
    std::cout << "ATTENTION: no wire nuber 1 for 4th layer!!!" << std::endl;
    LocalPoint dummyLP(0,0,0);
    return dummyLP;
  }
  const DTSuperLayer* sl   = _stat->superLayer(DTSuperLayerId(statId(),nsl));
  const DTLayer* lay       = sl->layer(DTLayerId(statId(),nsl,nlay));
  
   float localX             = lay->specificTopology().wirePosition(ntube);
   LocalPoint posInLayer(localX,0,0);
   LocalPoint posInChamber  = _stat->surface().toLocal(lay->toGlobal(posInLayer));
   //obsolete 19/06/2006 GlobalPoint  posInCMS = lay->toGlobal(posInLayer);
  
 /* cout <<endl;
  cout << "tube " << ntube << " nlay " << nlay << endl;
  cout << "posinlayer " << posInLayer << "posinchamb " << posInChamber << "posinCMS " << posInCMS << endl;*/
  
  return posInChamber;
}

int
DTTrigGeom::posFE(int sl) const {
   if( station()!=4 || sl!=2 ) {
     // obsolete 19/0602006 const DTLayer* lay  = _stat->superLayer(DTSuperLayerId(statId(),sl))->layer(DTLayerId(statId(),sl,1));
     return 1/*lay->getFEPosition()*/;                                               
   }
   else{
    std::cout << "No theta superlayer in station 4!" << std::endl;
    return 0;
  }
}

void
DTTrigGeom::getGeom() {

  // Geometrical constants of chamber
  // Cell width (cm)
  _PITCH = 4.2;
  // Cell height (cm)
  _H = 1.3;
  // azimuthal angle of normal to the chamber
  _PHICH = _stat->surface().toGlobal(LocalVector(0,0,-1)).phi();

  // superlayer positions and number of cells
  DTSuperLayer* sl[3];
  DTLayer* l1[3];
  DTLayer* l3[3];
  int i = 0;
  for(i=0; i<3; i++) {
    if(station()==4&&i==1) { // No theta SL in MB4
      _ZSL[i] = -999;
      _NCELL[i] = 0;
    } else {
      sl[i] = (DTSuperLayer*) _stat->superLayer(DTSuperLayerId(statId(),i+1));
      l1[i] = (DTLayer*) sl[i]->layer(DTLayerId(statId(),i+1,1));
      l3[i] = (DTLayer*) sl[i]->layer(DTLayerId(statId(),i+1,3));
      _ZSL[i] = _stat->surface().toLocal(sl[i]->position()).z(); // - 1.5 * _H;
      //LocalPoint posInLayer=l1[i]->layType()->getWire(1)->positionInLayer();
      const DTTopology& tp=l1[i]->specificTopology();
      float  posX=tp.wirePosition(tp.firstChannel());
      LocalPoint posInLayer(posX,0,0);
      LocalPoint posInChamber=_stat->surface().toLocal(l1[i]->surface().toGlobal(posInLayer));
      _NCELL[i] = l1[i]->specificTopology().channels();
    }
  }

  // debugging
  if(config()->debug()>3){
    std::cout << setiosflags(std::ios::showpoint | std::ios::fixed) << std::setw(4) <<
      std::setprecision(1);
    std::cout << "Identification: wheel=" << wheel();
    std::cout << ", station=" << station();
    std::cout << ", sector=" << sector() << std::endl;
    GlobalPoint pp = _stat->toGlobal(LocalPoint(0,0,0));
    std::cout << "Position: Mag=" << pp.mag() << "cm, Phi=" << pp.phi()*180/3.14159;
    std::cout << " deg, Z=" << pp.z() << " cm" << std::endl;
    std::cout << "Rotation: ANGLE=" << phiCh()*180/3.14159 << std::endl;
    //if(wheel()==2&&sector()==2){ // only 1 sector-wheel
      std::cout << "Z of superlayers: phi=" << ZSL(1) << ", ";
      std::cout << ZSL(3) << " theta=" << ZSL(2);
      std::cout << " (DeltaY = " << distSL() << ")" << std::endl;
      std::cout << " ncell: sl 1 " <<  nCell(1) << " sl 2 " <<  nCell(2) <<
              " sl 3 " <<  nCell(3) << std::endl;   
    //}
  }
  // end debugging

}

float 
DTTrigGeom::ZSL(int sl) const {
  if(sl<1||sl>3){
    std::cout << "DTTrigGeom::ZSL: wrong SL number: " << sl;
    std::cout << -999 << " returned" << std::endl;
    return -999;
  }
  return _ZSL[sl-1];
}


void 
DTTrigGeom::dumpGeom() const {
  std::cout << "Identification: wheel=" << wheel();
  std::cout << ", station=" << station();
  std::cout << ", sector=" << sector() << std::endl;
  GlobalPoint pp = _stat->toGlobal(LocalPoint(0,0,0));
  std::cout << "Position: Mag=" << pp.mag() << "cm, Phi=" << pp.phi()*180/3.14159;
  std::cout << " deg, Z=" << pp.z() << " cm" << std::endl;
  std::cout << "Rotation: ANGLE=" << phiCh()*180/3.14159 << std::endl;
  std::cout << "Z of superlayers: phi=" << ZSL(1) << ", ";
  std::cout << ZSL(3) << " theta=" << ZSL(2) << std::endl;
  std::cout << "Number of cells: SL1=" << nCell(1) << " SL2=" << nCell(2) <<
    " SL3=" << nCell(3) << std::endl;
  std::cout << "First wire positions:" << std::endl;
  int ii=0;
  int jj=0;
  for( ii = 1; ii<=3; ii++ ) {
    if(station()!=4||ii!=2){
      for ( jj =1; jj<=4; jj++ ) {
	std::cout << "    SL=" << ii << ", lay=" << jj << ", wire 1 position=";
        if ( jj ==4)
	  std::cout << tubePosInCh( ii, jj, 2) << std::endl;
	else
	  std::cout << tubePosInCh( ii, jj, 1) << std::endl;
      }
    }
  }
  std::cout << "First BTI position:";
  std::cout << " SL1:" << localPosition(DTBtiId(statId(),1,1));
  std::cout << " SL2:" << localPosition(DTBtiId(statId(),2,1));
  std::cout << " SL3:" << localPosition(DTBtiId(statId(),3,1)) << std::endl;
  std::cout << "First TRACO position:";
  std::cout << localPosition(DTTracoId(statId(),1)) << std::endl;
  std::cout << "******************************************************" << std::endl;
}


LocalPoint 
DTTrigGeom::localPosition(const DTBtiId id) const {
/* obsolete!
  float x = 0;
  float y = 0;
  float z = ZSL(id.superlayer());
  if(id.superlayer()==2){
    // SL 2: Reverse numbering -------V
    y = Xwire1BTI1SL(id.superlayer()) - ((float)(id.bti()-1)-0.5)*cellPitch();
  } else {
    x = Xwire1BTI1SL(id.superlayer()) + ((float)(id.bti()-1)-0.5)*cellPitch();
  }
*/

//NEWGEO
/*  int nsl = id.superlayer();
  int tube = mapTubeInFEch(nsl,1,id.bti());
  LocalPoint p = tubePosInCh(nsl,1,tube);
  //traslation because z axes is in middle of SL, x/y axes on left I of first cell
  
  LocalPoint p1 = tubePosInCh (nsl,1,1);
  LocalPoint p2 = tubePosInCh (nsl,2,1); 
  cout << "nbti " << id.bti() << " tube " << tube << " localpoint" << p << endl;
  cout << "localpoint layer 1" << p1  << " localpoint layer 2" << p2 << endl;
  
  float xt = 0;
  float yt = 0;
  float zt = - cellH() * 3./2.;
  if(nsl==2)
    yt = - cellPitch()/2.; 
  else
    xt = + cellPitch()/2.; 

  if(posFE(nsl)==0){//FE in positive y
      xt = - xt;
      yt = - yt;
  }
  
  cout << "localpoint " << p << ' '  << xt << ' ' << yt << endl;

  return LocalPoint(p.x()+xt,p.y()+yt,p.z()+zt);*/
	
	int nsl = id.superlayer();
	const DTSuperLayer* sl   = _stat->superLayer(DTSuperLayerId(statId(),nsl));
	const DTLayer* lay       = sl->layer(DTLayerId(statId(),nsl,1));
	int tube = id.bti();
	float localX             = lay->specificTopology().wirePosition(tube);
	float xt = -cellPitch()/2.;
	float zt = -cellH() * 3./2.;
	//LocalPoint posInLayer1(localX+xt,yt,0); //Correction now y is left I of first cell of layer 1 y=0 and z in the middle of SL,
	LocalPoint posInLayer1(localX+xt,0,zt);
	LocalPoint posInChamber  = _stat->surface().toLocal(lay->toGlobal(posInLayer1));
	//GlobalPoint posInCMS = lay->toGlobal(posInLayer1);

	/* cout <<endl;
	cout << "tube " << ntube << " nlay " << nlay << endl;
	cout << "posinlayer " << posInLayer1 << "posinchamb " << posInChamber << "posinCMS " << posInCMS << endl;*/
	
	return posInChamber;
}

LocalPoint 
DTTrigGeom::localPosition(const DTTracoId id) const {
/* obsolete
  float x = Xwire1BTI1SL(1) +
    ( ( (float)(id.traco()) - 0.5 ) * DTConfig::NBTITC - 0.5 )*cellPitch();
  // half cell shift in SL1 of MB1 (since cmsim116)
  if(station()==1) x -= 0.5*cellPitch();
  float y = 0;
  float z = ZcenterSL();
*/
  //NEWGEO
  // position of first BTI in sl 3 on X
  float x = localPosition( DTBtiId(DTSuperLayerId(wheel(),station(),sector(),3),1) ).x();
// 10/7/06 May be not needed anymore in new geometry
//   if(posFE(3)==1)
//     x -= (id.traco()-2)*DTConfig::NBTITC * cellPitch();
//   if(posFE(3)==0)
    x += (id.traco()-2)*DTConfig::NBTITC * cellPitch();

  float y = 0;
  float z = ZcenterSL();

  return LocalPoint(x,y,z);
}

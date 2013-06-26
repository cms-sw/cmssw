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
//   S. Vanini 090902 : dumpLUT method implemented
//   A. Gozzelino May 11th 2012: IEEE32toDSP method bug fix
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
#include <fstream>
#include <sstream>
#include <cstring>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

using namespace std;

//----------------
// Constructors --
//----------------

DTTrigGeom::DTTrigGeom(DTChamber* stat, bool debug) : _stat(stat) , _debug(debug) {

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
DTTrigGeom::setGeom(const DTChamber* stat) {

  _stat=stat;
  getGeom();

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
  int i = 0;
  for(i=0; i<3; i++) {
    if(station()==4&&i==1) { // No theta SL in MB4
      _ZSL[i] = -999;
      _NCELL[i] = 0;
    } else {
      sl[i] = (DTSuperLayer*) _stat->superLayer(DTSuperLayerId(statId(),i+1));
      l1[i] = (DTLayer*) sl[i]->layer(DTLayerId(statId(),i+1,1));
      _ZSL[i] = _stat->surface().toLocal(sl[i]->position()).z(); // - 1.5 * _H;
      //LocalPoint posInLayer=l1[i]->layType()->getWire(1)->positionInLayer();
      const DTTopology& tp=l1[i]->specificTopology();
      float  posX=tp.wirePosition(tp.firstChannel());
      LocalPoint posInLayer(posX,0,0);
      _NCELL[i] = l1[i]->specificTopology().channels();
    }
  }

  // debugging
  if(_debug){
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

  GlobalPoint gp1 = CMSPosition(DTBtiId(statId(),1,1)); 
  

  std::cout << "First BTI position:";
  std::cout << " SL1:" << localPosition(DTBtiId(statId(),1,1)) << std::endl;
  std::cout << " Position: R=" << gp1.perp() << "cm, Phi=" << gp1.phi()*180/3.14159 << " deg, Z=" << gp1.z() << " cm" << std::endl;

  if(station()!=4)
  {
	GlobalPoint gp2 = CMSPosition(DTBtiId(statId(),2,1)); 
	std::cout << " SL2:" << localPosition(DTBtiId(statId(),2,1))<< std::endl;
	std::cout << " Position: R=" << gp2.perp() << "cm, Phi=" << gp2.phi()*180/3.14159 << " deg, Z=" << gp2.z() << " cm" << std::endl;
  }

  GlobalPoint gp3 = CMSPosition(DTBtiId(statId(),3,1)); 
  std::cout << " SL3:" << localPosition(DTBtiId(statId(),3,1)) << std::endl;
  std::cout << " Position: R=" << gp3.perp() << "cm, Phi=" << gp3.phi()*180/3.14159 << " deg, Z=" << gp3.z() << " cm" << std::endl;

  std::cout << "First TRACO position:";
  std::cout << localPosition(DTTracoId(statId(),1)) << std::endl;
  std::cout << "******************************************************" << std::endl;
}

void 
DTTrigGeom::dumpLUT(short int btic) {

  // chamber id
  int wh = wheel();
  int st = station();
  int se = sector();

  // open txt file 
  string name = "Lut_from_CMSSW_geom";
 /* name += "_wh_";
  if(wh<0)
	name += "-";
  name += abs(wh) + '0';
  name += "_st_";
  name += st + '0';
  name += "_se_";
  if(se<10)
	name += se + '0';
  else 
  {
	name += 1 + '0';
	name += (se-10) + '0';
  }
  */ 
  name += ".txt";

  ofstream fout;
  fout.open(name.c_str(),ofstream::app);

// *** dump file header
//  fout << "Identification: wheel\t" << wh;
//  fout << "\tstation\t" << st;
//  fout << "\tsector\t" << se;
  fout << wh;
  fout << "\t" << st;
  fout << "\t" << se;

  // SL shift
  float xBTI1_3 	= localPosition( DTBtiId(DTSuperLayerId(wheel(),station(),sector(),3),1) ).x();
  float xBTI1_1 	= localPosition( DTBtiId(DTSuperLayerId(wheel(),station(),sector(),1),1) ).x();
  float SL_shift 	= xBTI1_3 - xBTI1_1;
  //  std::cout << " SL shift " << SL_shift << std::endl;

  // traco 1 and 2 global position
  LocalPoint traco1 	= localPosition(DTTracoId(statId(),1));
  LocalPoint traco2 	= localPosition(DTTracoId(statId(),2));
  GlobalPoint traco_1 	= toGlobal(traco1);
  GlobalPoint traco_2 	= toGlobal(traco2);
  // std::cout << " tr1 x " << traco_1.x() << " tr2 x " << traco_2.x() << std::endl;

  float d;
  float xcn;
  int xcn_sign;
  GlobalPoint pp = _stat->toGlobal(LocalPoint(0,0,ZcenterSL()));
  // std::cout << "Position: x=" << pp.x() << "cm, y=" << pp.y() << "cm, z=" << pp.z() << std::endl;  
    
  if(sector()==1 || sector() ==7){  
  	d = fabs(traco_1.x());
  	xcn = fabs(traco_1.y());
        // 110208 SV comment: this was inserted for a TRACO hardware bug
  	if (SL_shift > 0) 
		xcn = xcn+SL_shift;
  	xcn_sign = static_cast<int>(pp.y()/fabs(pp.y()))*static_cast<int>(traco_1.y()/fabs(traco_1.y()));
  	if(station() == 2 || (station() == 4 && sector() == 1)) 
		xcn_sign = - xcn_sign;
  	xcn = xcn*xcn_sign;
  }
  else {
  	float m1 = (traco_2.y()-traco_1.y())/(traco_2.x()-traco_1.x());
  	float q1 = traco_1.y()-m1*traco_1.x();
  	float m = tan(phiCh());
  	float xn = q1/(m-m1);
  	float yn = m*xn;
  
  	d = sqrt(xn*xn+yn*yn);
  	xcn = sqrt( (xn-traco_1.x())*(xn-traco_1.x()) + (yn-traco_1.y())*(yn-traco_1.y()) );
        // 110208 SV comment: this was inserted for a TRACO hardware bug
  	if (SL_shift > 0) 
		xcn = xcn+SL_shift;
  
  	float diff = (pp.x()-traco_1.x())*traco_1.y();
  	xcn_sign = static_cast<int>(diff/fabs(diff));
  	xcn = xcn*xcn_sign;
  }
  // std::cout << " d " << d << " xcn " << xcn << " sign " << xcn_sign << std::endl; 
  //fout << "\td\t" << d << "\txcn\t" << xcn << "\t"; 
  //fout << "btic\t" << btic << "\t";

// *** dump TRACO LUT command
  fout << "\tA8";
  //short int btic = 31;
  //cout << "CHECK BTIC " << btic << endl;
  short int Low_byte = (btic & 0x00FF);   // output in hex bytes format with zero padding
  short int High_byte =( btic>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0') << Low_byte;	
    
  // convert parameters from IEE32 float to DSP float format
  short int DSPmantissa = 0;
  short int DSPexp = 0;

  // d parameter conversion and dump
  IEEE32toDSP(d, DSPmantissa, DSPexp);
  Low_byte = (DSPmantissa & 0x00FF);   // output in hex bytes format with zero padding
  High_byte =( DSPmantissa>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0') << Low_byte;	
  Low_byte = (DSPexp & 0x00FF);
  High_byte =( DSPexp>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << High_byte << setw(2) << setfill('0') << Low_byte;	

  // xnc parameter conversion and dump
  DSPmantissa = 0;
  DSPexp = 0;
  IEEE32toDSP(xcn, DSPmantissa, DSPexp);
  Low_byte = (DSPmantissa & 0x00FF);   // output in hex bytes format with zero padding
  High_byte =( DSPmantissa>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0') << Low_byte;	
  Low_byte = (DSPexp & 0x00FF);
  High_byte =( DSPexp>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << High_byte << setw(2) << setfill('0') << Low_byte;	

  // sign bits 
  Low_byte = (xcn_sign & 0x00FF);   // output in hex bytes format with zero padding
  High_byte =( xcn_sign>>8 & 0x00FF);
  fout << setw(2) << setfill('0') << hex << High_byte << setw(2) << setfill('0') << Low_byte << dec << "\n"; 

  fout.close();

  return;
 
}

/* 
// A. Gozzelino May 11th 2012: Old and wrong definition
void 
DTTrigGeom::IEEE32toDSP(float f, short int & DSPmantissa, short int & DSPexp)
{
  long int *pl=0, lm;
  bool sign=false;

  DSPmantissa = 0;
  DSPexp = 0;

  if( f!=0.0 )
  {
	memcpy(pl,&f,sizeof(float));
		
        if((*pl & 0x80000000)!=0) 
		sign=true;	  
        lm = ( 0x800000 | (*pl & 0x7FFFFF)); // [1][23bit mantissa]
        lm >>= 9; //reduce to 15bits
	lm &= 0x7FFF;
        DSPexp = ((*pl>>23)&0xFF)-126;
	DSPmantissa = (short)lm;
	if(sign) 
		DSPmantissa = - DSPmantissa;  // convert negative value in 2.s complement	

  }
  return;
}
*/

//*******************
// A.Gozzelino May 11th 2012: bug fix in method IEEE32toDSP
//******************

void 
DTTrigGeom::IEEE32toDSP(float f, short int & DSPmantissa, short int & DSPexp)
{  
  long int lm;
  long int pl = 0;
  
  bool sign=false;

  DSPmantissa = 0;
  DSPexp = 0;

  if( f!=0.0 )
  {
	memcpy(&pl,&f,sizeof(float));
	
        if((pl & 0x80000000)!=0) 
		sign=true;	  
        lm = ( 0x800000 | (pl & 0x7FFFFF)); // [1][23bit mantissa]
        lm >>= 9; //reduce to 15bits
	lm &= 0x7FFF;
        DSPexp = ((pl>>23)&0xFF)-126;
	DSPmantissa = (short)lm;
	if(sign) 
		DSPmantissa = - DSPmantissa;  // convert negative value in 2.s complement	

  }
  return;
}
//********************** end bug fix ****************

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

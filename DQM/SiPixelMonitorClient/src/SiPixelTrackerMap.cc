// -*- C++ -*-
//
// Package:    SiPixelMonitorClient
// Class:      SiPixelTrackerMap
// 
/**\class 

 Description: Pixel DQM source used to generate an svgmap.xml file with the topology of 
 the Pixel Detector (used by web-based clients)

 Implementation:
     This class inherits from the Common/Tools/TrackerMap.cc and reimplements the methods used 
     to create a reduced-set map with Pixel Detectors only
     
*/
//
// Original Author:  Dario Menasce
//         Created:  
// $Id: SiPixelTrackerMap.cc,v 1.14 2009/06/18 14:04:42 merkelp Exp $
//
//
#include "DQM/SiPixelMonitorClient/interface/SiPixelTrackerMap.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelContinuousPalette.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"

#include "TText.h"

#include <fstream>
#include <iostream>
using namespace std;

//----------------------------------------------------------------------------------------------------
SiPixelTrackerMap::SiPixelTrackerMap(string s,int xsize1, int ysize1) : TrackerMap(s,xsize1,ysize1) 
{
// cout << ACBlue << ACBold
//      << "[SiPixelTrackerMap::SiPixelTrackerMap()]" 
//      << endl ;
  title = s ;
 // cout<<"created a new Tracker Map! the title is: "<<s<<endl;
}
		  
//----------------------------------------------------------------------------------------------------
void SiPixelTrackerMap::drawModule(TmModule * mod, int key,int nlay, bool print_total)
{
  //int x,y;
  double phi,r,dx,dy, dy1;
  double xp[4],yp[4],xp1,yp1;
  double vhbot,vhtop,vhapo;
  double rmedio[]={0.041,0.0701,0.0988,0.255,0.340,0.430,0.520,0.610,0.696,0.782,0.868,0.965,1.080};
  double xt1,yt1,xs1=0.,ys1=0.,xt2,yt2,xs2,ys2,pv1,pv2;
//  int green = 0;
  double xd[4],yd[4];
  int np = 4;
  //int numrec=0;
  int numod=0;
  phi = phival(mod->posx,mod->posy);
  r = sqrt(mod->posx*mod->posx+mod->posy*mod->posy);
  vhbot = mod->width;
  vhtop=mod->width;
  vhapo=mod->length;
  if(nlay < 31){ //endcap
    vhbot = mod->widthAtHalfLength/2.-(mod->width/2.-mod->widthAtHalfLength/2.); 
    vhtop=mod->width/2.;
    vhapo=mod->length/2.;
    if(nlay >12 && nlay <19){
      if(posrel)r = r+r;
      xp[0]=r-vhtop;yp[0]=-vhapo;
      xp[1]=r+vhtop;yp[1]=-vhapo;
      xp[2]=r+vhtop;yp[2]=vhapo;
      xp[3]=r-vhtop;yp[3]=vhapo;
    }else{
      if(posrel)r = r + r/3.;
      xp[0]=r-vhapo;yp[0]=-vhbot;
      xp[1]=r+vhapo;yp[1]=-vhtop;
      xp[2]=r+vhapo;yp[2]=vhtop;
      xp[3]=r-vhapo;yp[3]=vhbot;
    }
    for(int j=0;j<4;j++){
      xp1 = xp[j]*cos(phi)-yp[j]*sin(phi);
      yp1 = xp[j]*sin(phi)+yp[j]*cos(phi);
      xp[j] = xp1;yp[j]=yp1;
    }
  } else { //barrel
    numod=mod->idModule;if(numod>100)numod=numod-100;
    int vane = mod->ring;
    if(posrel){
      dx = vhapo;
      phi=M_PI;
      xt1=rmedio[nlay-31]; yt1=-vhtop/2.;
      xs1 = xt1*cos(phi)-yt1*sin(phi);
      ys1 = xt1*sin(phi)+yt1*cos(phi);
      xt2=rmedio[nlay-31]; yt2=vhtop/2.;
      xs2 = xt2*cos(phi)-yt2*sin(phi);
      ys2 = xt2*sin(phi)+yt2*cos(phi);
      dy=phival(xs2,ys2)-phival(xs1,ys1);
	 dy1 = dy;
      if(nlay==31)dy1=0.39;
      if(nlay==32)dy1=0.23;
      if(nlay==33)dy1=0.16;
      xp[0]=vane*(dx+dx/8.);yp[0]=numod*(dy1);
      xp[1]=vane*(dx+dx/8.)+dx;yp[1]=numod*(dy1);
      xp[2]=vane*(dx+dx/8.)+dx;yp[2]=numod*(dy1)+dy;
      xp[3]=vane*(dx+dx/8.);yp[3]=numod*(dy1)+dy;
    }else{
      xt1=r; yt1=-vhtop/2.;
      xs1 = xt1*cos(phi)-yt1*sin(phi);
      ys1 = xt1*sin(phi)+yt1*cos(phi);
      xt2=r; yt2=vhtop/2.;
      xs2 = xt2*cos(phi)-yt2*sin(phi);
      ys2 = xt2*sin(phi)+yt2*cos(phi);
      pv1=phival(xs1,ys1);
      pv2=phival(xs2,ys2);
      if(fabs(pv1-pv2)>M_PI && numod==1)pv1=pv1-2.*M_PI;
      if(fabs(pv1-pv2)>M_PI && numod!=1)pv2=pv2+2.*M_PI;
      xp[0]=mod->posz-vhapo/2.;yp[0]=4.2*pv1;
      xp[1]=mod->posz+vhapo/2.;yp[1]=4.2*pv1;
      xp[2]=mod->posz+vhapo/2. ;yp[2]=4.2*pv2;
          xp[3]=mod->posz-vhapo/2.;yp[3]=4.2*pv2;
    }
  }
  if(isRingStereo(key))
        {
	  np = 3;
	  if(mod->idModule>100 ){for(int j=0;j<3;j++){
	      xd[j]=xdpixel(xp[j]);yd[j]=ydpixel(yp[j]);
	    }
	  }else {
	    xd[0]=xdpixel(xp[2]);yd[0]=ydpixel(yp[2]);
	    xd[1]=xdpixel(xp[3]);yd[1]=ydpixel(yp[3]);
	    xd[2]=xdpixel(xp[0]);yd[2]=ydpixel(yp[0]);
	  }
        } else {
    for(int j=0;j<4;j++){
      xd[j]=xdpixel(xp[j]);yd[j]=ydpixel(yp[j]);
    }
  }
  char buffer [20];
  sprintf(buffer,"%X",mod->idex);

//cout<<"drawModule: xp= "<<xp<<" , yp= "<<yp<<endl;
  bool FPIX_M_1 = false ;
  bool FPIX_M_2 = false ;
  bool FPIX_P_1 = false ;
  bool FPIX_P_2 = false ;
  bool BPIX_L_1 = false ;
  bool BPIX_L_2 = false ;
  bool BPIX_L_3 = false ;
  string moduleName = mod->name;
  if(moduleName.find("PixelEndcap")!=string::npos || moduleName.find("PixelBarrel")!=string::npos) {
    FPIX_M_1 = false ;
    FPIX_M_2 = false ;
    FPIX_P_1 = false ;
    FPIX_P_2 = false ;
    BPIX_L_1 = false ;
    BPIX_L_2 = false ;
    BPIX_L_3 = false ;
    if( moduleName.find("PixelEndcap 3")!=string::npos ) {FPIX_M_1 = true;}
    if( moduleName.find("PixelEndcap 4")!=string::npos ) {FPIX_M_2 = true;}
    if( moduleName.find("PixelEndcap 1")!=string::npos ) {FPIX_P_1 = true;}
    if( moduleName.find("PixelEndcap 2")!=string::npos ) {FPIX_P_2 = true;}
    if( moduleName.find("PixelBarrel 1")!=string::npos ) {BPIX_L_1 = true;}
    if( moduleName.find("PixelBarrel 2")!=string::npos ) {BPIX_L_2 = true;}
    if( moduleName.find("PixelBarrel 3")!=string::npos ) {BPIX_L_3 = true;}
   //}
   *svgfile << "      <svg:polygon detid=\"" 
  	    << mod->idex
  	    << "\" id=\""
  	    << mod->idex
  	    << "\" onclick=\"SvgMap.showData(evt);\" onmouseover=\"SvgMap.showData(evt);\" onmouseout=\"SvgMap.showData(evt);\" entries=\""
  	    << mod->text
  	    << "\" POS=\""
  	    << mod->name
  	    << " Id "
  	    << mod->idex
  	    << " \" fill=\"rgb("
  	    << 146
  	    << ","
  	    << 0
  	    << ","
  	    << 255
  	    << ")\" points=\"";
   for(int k=0;k<np;k++)
   {
    if( FPIX_M_1 )
    {
     xd[k] = xd[k] * 1.8 -   60 ;
     yd[k] = yd[k] * 2.0 -   30 ;
    }
    if( FPIX_M_2 )
    {
     xd[k] = xd[k] * 1.8 -   60 ;
     yd[k] = yd[k] * 2.0 -   60 ;
    }
    if( FPIX_P_1 )
    {
     xd[k] = xd[k] * 1.8 - 1020 ;
     yd[k] = yd[k] * 2.0 -   30 ;
    }
    if( FPIX_P_2 )
    {
     xd[k] = xd[k] * 1.8 - 1020 ;
     yd[k] = yd[k] * 2.0 -   60 ;
    }
    if( BPIX_L_1 ) 
    { 
     xd[k] = xd[k] * 1.2 -  130 ;
    }
    if( BPIX_L_2 )
    {
     xd[k] = xd[k] * 1.2 -   30 ;
    }
    if( BPIX_L_3 )
    {
     xd[k] = xd[k] * 1.2 -  240 ;
     yd[k] = yd[k]       -    5 ;
    }
    *svgfile << xd[k] << "," << yd[k] << " " ;
   }
   *svgfile <<"\" />" <<endl;
   return ;
  }

}

//===================================================================================
//print in svg format tracker map
//print_total = true represent in color the total stored in the module
//print_total = false represent in color the average  
void SiPixelTrackerMap::print(bool print_total, string TKType, float minval, float maxval)
{
//cout<<"Entering SiPixelTrackerMap::print: "<<endl;

 minvalue=minval; maxvalue=maxval;
 svgfile = new ofstream("svgmap.xml",ios::out);
 jsfile = new ifstream("TrackerMapHeader.txt",ios::in);
 
 //copy javascript interface from trackermap.txt file
 string line;
 while (getline( *jsfile, line ))
 {
     *svgfile << line << endl;
 }
 //

 if(!print_total)
 {
  for (int layer=1; layer < 44; layer++)
  {
   for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++)
   {
    for (int module=1;module<200;module++) 
    {
     int key=layer*100000+ring*1000+module;
     TmModule * mod = smoduleMap[key];
     if(mod !=0 && !mod->notInUse())
     {
       mod->value = mod->value / mod->count;
     }
    }
   }
  }
 }
 
 if(minvalue>=maxvalue)
 {
  minvalue=9999999.;
  maxvalue=-9999999.;
  for (int layer=1; layer < 44; layer++)
  {
   for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++)
   {
    for (int module=1;module<200;module++) 
    {
     int key=layer*100000+ring*1000+module;
     TmModule * mod = smoduleMap[key];
     if(mod !=0 && !mod->notInUse())
     {
       if (minvalue > mod->value)minvalue=mod->value;
       if (maxvalue < mod->value)maxvalue=mod->value;
     }
    }
   }
  }
 }
 
 for (int layer=1; layer < 44; layer++)
 {
  nlay=layer;
  defwindow(nlay);
  for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++)
  {
   for (int module=1;module<200;module++) 
   {
    int key=layer*100000+ring*1000+module;
    TmModule * mod = smoduleMap[key];
    if(mod !=0 && !mod->notInUse())
    {
      drawModule(mod,key,layer,print_total);
    }
   }
  }
 }

/*  cout << ACYellow << ACBold
       << "[SiPixelTrackerMap::print()] "
       << ACPlain
       << "TKType: |" 
       << TKType
       << "|"
       << endl ;
*/
  *svgfile << "  " << endl ;

 if( TKType == "Averages" || TKType == "Entries")
 {
  *svgfile << "      <svg:g id=\"theColorMap\" transform=\"translate(0, 0)\" style=\"visibility: visible;\">" << endl ;
 } else {
  *svgfile << "      <svg:g id=\"theColorMap\" transform=\"translate(0, 0)\" style=\"visibility: hidden;\">" << endl ;
 }
 
 // this is the color scale on the right hand side of the tracker map:
 int px = 1370 ;
 int dx =   25 ;
 int py =   50 ;
 int dy =    6 ;
 int  j =    5 ;
 for( int i=99; i>=0; i--)
 {
  *svgfile << "       <svg:polygon id=\"map\" fill=\"rgb("
 	   << SiPixelContinuousPalette::r[i]
 	   << ","
 	   << SiPixelContinuousPalette::g[i]
 	   << ","
 	   << SiPixelContinuousPalette::b[i]
 	   << ")\" points=\""
 	   << px
 	   << ","
 	   << py
 	   << " "
 	   << px+dx
 	   << ","
 	   << py
 	   << " "
 	   << px+dx
 	   << ","
 	   << py+dy
 	   << " "
 	   << px
 	   << ","
 	   << py+dy
 	   << "\" />"
 	   << endl ;
  if( i == 0 || i==20 || i==40 || i==60 || i==80 || i==99) 
  {
   *svgfile << "  " << endl ;
   *svgfile << "       <svg:text id=\"colorCodeMark"
 	    << j-- 
 	    << "\" class=\"normalText\" x=\""
 	    << px+dx+5
 	    << "\" y=\""
 	    << py + 3
 	    << "\" font-size=\"20\">"
 	    << i
 	    << "%</svg:text>"
 	    << endl;
   *svgfile << "  " << endl ;
  }
  py += dy + 1;
  
 // cout<<"inside the polygon loop: i= "<<i<<" , r= "<<SiPixelContinuousPalette::r[i]<<" , g= "<<SiPixelContinuousPalette::g[i]<<" , b= "<<SiPixelContinuousPalette::b[i]<<" , px= "<<px<<" , py= "<<py<<endl;
 }
 *svgfile << "      </svg:g>" << endl ;

 if( TKType == "Alarms" )
 {
  *svgfile << "      <svg:g id=\"theAlarmMap\" transform=\"translate(0, 0)\" style=\"visibility: visible;\">" << endl ;
 } else {
  *svgfile << "      <svg:g id=\"theAlarmMap\" transform=\"translate(0, 0)\" style=\"visibility: hidden;\">" << endl ;
 }
 *svgfile << " "															  << endl ;
 *svgfile << "       <svg:polygon id=\"map\"	  fill =\"rgb(255,0,0)\"     points=\"1300,300  1325,300  1325,320  1300,320\"  />"	  << endl ;
 *svgfile << "       <svg:text    id=\"ERROR\"     class=\"normalText\"      x=\"1334\" y=\"317\" font-size=\"20\">ERROR    </svg:text>"  << endl ;
 *svgfile << " "															  << endl ;  
 *svgfile << "       <svg:polygon id=\"map\"	  fill =\"rgb(0,255,0)\"     points=\"1300,330  1325,330  1325,350  1300,350\" />"	  << endl ;
 *svgfile << "       <svg:text    id=\"OK\"	  class=\"normalText\"       x=\"1334\" y=\"347\" font-size=\"20\">OK	     </svg:text>" << endl ;
 *svgfile << " "															  << endl ;  
 *svgfile << "       <svg:polygon id=\"map\"	  fill =\"rgb(255,255,0)\"   points=\"1300,360  1325,360  1325,380  1300,380\" />"	  << endl ;
 *svgfile << "       <svg:text    id=\"WARNING\"   class=\"normalText\"      x=\"1334\" y=\"377\" font-size=\"20\">WARNING  </svg:text>"  << endl ;
 *svgfile << " "															  << endl ;  
 *svgfile << "       <svg:polygon id=\"map\"	  fill =\"rgb(0,0,255)\"     points=\"1300,390  1325,390  1325,410  1300,410\" />"	  << endl ;
 *svgfile << "       <svg:text    id=\"OTHER\"     class=\"normalText\"      x=\"1334\" y=\"407\" font-size=\"20\">OTHER    </svg:text>"  << endl ;
 *svgfile << " "															  << endl ;  
 *svgfile << "       <svg:polygon id=\"map\"	  fill =\"rgb(255,255,255)\" points=\"1300,420  1325,420  1325,440  1300,440\" />"	  << endl ;
 *svgfile << "       <svg:text    id=\"UNDEFINED\" class=\"normalText\"      x=\"1334\" y=\"437\" font-size=\"20\">UNDEFINED</svg:text>"  << endl ;
 *svgfile << " "															  << endl ;  
 *svgfile << "      </svg:g>"														  << endl ;

 *svgfile << " " << endl ;

 *svgfile << "      <svg:text id=\"colorCodeME\" class=\"normalText\" x=\"1000\" y=\"4000\">"
 	  << title
          << "</svg:text>"
          << endl;
 delete jsfile ;					 
 jsfile = new ifstream("TrackerMapTrailer.txt",ios::in); 
 while (getline( *jsfile, line ))			 
       {						 
 	   *svgfile << line << endl;			 
       }						 
 delete jsfile ;
 
 svgfile->close() ;
/*  cout << ACYellow << ACBold
       << "[SiPixelTrackerMap::print(  )] "
       << ACPlain
       << "svgmap.xml file just closed..."
       << endl ;
*/ 					 
 delete svgfile ;					 
}

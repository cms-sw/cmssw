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
// $Id: SiPixelTrackerMap.cc,v 1.0 2007/05/18 20:24:44 menasce Exp $
//
//
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelTrackerMap.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelContinuousPalette.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"

#include <qstring.h>
#include <qregexp.h>

#include <fstream>
#include <iostream>
using namespace std;

SiPixelTrackerMap::SiPixelTrackerMap(string s,int xsize1, int ysize1) : TrackerMap(s,xsize1,ysize1) 
{
// cout << ACBlue << ACBold
//      << "[SiPixelTrackerMap::SiPixelTrackerMap()]" 
//      << endl ;
  title = s ;
}
		  
void SiPixelTrackerMap::drawModule(TmModule * mod, int key,int nlay, bool print_total){
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

  QRegExp rx("(BPIX|FPIX)") ;
  QString modName = mod->name ;
  if( rx.search(modName) != -1 )
  {
   *svgfile << "      <svg:polygon detid=\"" 
  	    << mod->idex
  	    << "\" id=\""
  	    << mod->idex
  	    << "\" onclick=\"showData(evt);\" onmouseover=\"showData(evt);\" onmouseout=\"showData(evt);\" MESSAGE=\""
  	    << mod->text
  	    << "\" POS=\""
  	    << mod->name
  	    << " Id "
  	    << mod->idex
  	    << " \" fill=\"rgb("
  	    << mod->red
  	    << ","
  	    << mod->green
  	    << ","
  	    << mod->blue
  	    << ")\" points=\"";
   for(int k=0;k<np;k++)
   {
    *svgfile << xd[k] << "," << yd[k] << " " ;
   }
   *svgfile <<"\" />" <<endl;
   return ;
  }

}
//print in svg format tracker map
//print_total = true represent in color the total stored in the module
//print_total = false represent in color the average  
void SiPixelTrackerMap::print(bool print_total, float minval, float maxval)
{
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

 if(!print_total){
  for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = SvgModuleMap::smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
          mod->value = mod->value / mod->count;
        }
      }
    }
  }
  }
  if(minvalue>=maxvalue){
  minvalue=9999999.;
  maxvalue=-9999999.;
  for (int layer=1; layer < 44; layer++){
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = SvgModuleMap::smoduleMap[key];
        if(mod !=0 && !mod->notInUse()){
          if (minvalue > mod->value)minvalue=mod->value;
          if (maxvalue < mod->value)maxvalue=mod->value;
        }
      }
    }
  }
}
  for (int layer=1; layer < 44; layer++){
    nlay=layer;
    defwindow(nlay);
    for (int ring=firstRing[layer-1]; ring < ntotRing[layer-1]+firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
        int key=layer*100000+ring*1000+module;
        TmModule * mod = SvgModuleMap::smoduleMap[key];
	if(mod !=0 && !mod->notInUse()){
          drawModule(mod,key,layer,print_total);
        }
      }
    }
  }

  *svgfile << "  " << endl ;
  *svgfile << "      <svg:g id=\"theColorMap\" transform=\"translate(0, 0)\">" << endl ;
  int px = 1350 ;
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
  }
  *svgfile << "      </svg:g>" << endl ;
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
  delete svgfile ;					  
}

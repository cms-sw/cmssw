// Root macro to create the 41 images of a Tracker Scan
// Code developed with the help of Raffaella Radogna physics student of 
//  Bari University
#include <map>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "TCanvas.h"
#include "TPad.h"
#include "TPolyLine.h"
#include "TPolyMarker.h"
#include <map>
#include "TROOT.h"


bool posrel;
bool saveAsSingleLayer;
int nlay,key;
double xmin,xmax,ymin,ymax;
 int xsize,ysize,ix,iy;
 int cwidth,cheight;
TCanvas * MyGC;

struct coordinateModulo{
  int part;
  int subpart;
  int layer;
  int ring;
  int nmod;
  double posx, posy, posz;
  double length,  width,  thickness, widthAtHalfLength;
  int detid;
  std::string nome; 
} mod[16588];

void defwindow(int num_lay){
  nlay = num_lay;
  if(posrel){ // separated modules
    xmin=-2.;ymin=-2.;xmax=2.;ymax=2.;
    if(nlay >12 && nlay < 19){
      xmin=-.40;xmax=.40;ymin=-.40;ymax=.40;
    }
    if(nlay>30){
      xmin=-0.1;xmax=3.;ymin=-0.1;ymax=8.5;
      if(nlay<34){xmin=-0.3;xmax=1.0;}
      if(nlay>33&&nlay<38){xmax=2.0;}
      if(nlay>37){ymax=8.;}//inner
    }
  }else{ //overlayed modules
    xmin=-1.3;ymin=-1.3;xmax=1.3;ymax=1.3;
    if(nlay >12 && nlay < 19){
      xmin=-.20;xmax=.20;ymin=-.20;ymax=.20;
    }
    if(nlay>30){
      xmin=-1.5;xmax=1.5;ymin=-1.;ymax=28.;
      if(nlay<34){xmin=-0.5;xmax=0.5;}
      if(nlay>33&&nlay<38){xmin=-1.;xmax=1.;}
    }
    
  }
  if(nlay<16){
    ix=0;iy=(15-nlay)*ysize;}
  if(nlay>15&&nlay<31){
    ix=3*xsize;iy=(nlay-16)*ysize;}
  if(nlay>30){
    if(nlay==31){ix=(int)(1.5*xsize);iy=0;}
    if(nlay>31 && nlay%2==0){int il=(nlay-30)/2;ix=xsize;iy=il*2*ysize;}
    if(nlay>31 && nlay%2!=0){int il=(nlay-30)/2;ix=2*xsize;iy=il*2*ysize;}
  }
 }

double  xdpixel(double x){
  double res;
  if(saveAsSingleLayer)res= ((x-xmin)/(xmax-xmin)*xsize);
  else res= ((x-xmin)/(xmax-xmin)*xsize)+ix;
  return res;
}

double  ydpixel(double y){
  double res;
  double y1;
  y1 = (y-ymin)/(ymax-ymin);
  if(nlay>30)   res= 2*ysize - (y1*2*ysize);
  else res= ysize - (y1*ysize);
  if(!saveAsSingleLayer) res=res+iy;
  return res;
}

double phival(double x, double y){
  double phi;
  double phi1=atan(y/x);
  phi = phi1;
  if(y<0. && x>0) phi = phi1+2.*M_PI;
  if(x<0.)phi=phi1+M_PI;
  if(fabs(y)<0.000001 && x>0)phi=0;
  if(fabs(y)<0.000001&&x<0)phi=M_PI;
  if(fabs(x)<0.000001&&y>0)phi=M_PI/2.;
  if(fabs(x)<0.000001&&y<0)phi=3.*M_PI/2.;
  return phi;
}
bool isRingStereo(int key){
    int layer=key/100000;
    int ring = key - layer*100000;
    ring = ring/1000;
    if(layer==34 || layer==35 || layer==38 || layer==39) return true;
    if(layer<13 || (layer>18&&layer<31))
      if(ring==1 || ring==2 || ring==5)return true;
    return false;
  }


void computemodule(int modulo,int nlay,int *npoints,double *xpol, double *ypol ){
  double phi,r;
  double xp[4],yp[4],xp1,yp1;
  double vhbot,vhtop,vhapo;
  double xt1,yt1,xs1=0.,ys1=0.,xt2,yt2,xs2,ys2,pv1,pv2;
  *npoints=5; 
  phi = phival(mod[modulo].posx,mod[modulo].posy);
  r = sqrt(mod[modulo].posx*mod[modulo].posx+mod[modulo].posy*mod[modulo].posy);
  vhbot = mod[modulo].width; 	  
  vhtop=mod[modulo].width; 	  
  vhapo=mod[modulo].length;
  if(nlay < 31){//endcap
    vhbot = mod[modulo].widthAtHalfLength/2.-(mod[modulo].width/2.-mod[modulo].widthAtHalfLength/2.); 	  
    vhtop=mod[modulo].width/2.; 	  
    vhapo=mod[modulo].length/2.;
    if(nlay >12 && nlay <19){//pix endcap
      xp[0]=r-vhtop;yp[0]=-vhapo;
      xp[1]=r+vhtop;yp[1]=-vhapo;
      xp[2]=r+vhtop;yp[2]=vhapo;
      xp[3]=r-vhtop;yp[3]=vhapo;
    }else{
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
    int numod;
    numod=mod[modulo].nmod;if(numod>100)numod=numod-100;
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
      xp[0]=mod[modulo].posz-vhapo/2.;yp[0]=4.2*pv1;
      xp[1]=mod[modulo].posz+vhapo/2.;yp[1]=4.2*pv1;
      xp[2]=mod[modulo].posz+vhapo/2. ;yp[2]=4.2*pv2;
      xp[3]=mod[modulo].posz-vhapo/2.;yp[3]=4.2*pv2;
  }
  for(int j=0;j<4;j++){
    ypol[j]=xdpixel(xp[j]);xpol[j]=ydpixel(yp[j]);
  }
  ypol[4]=ypol[0];xpol[4]=xpol[0];
  //for(int j=0; j<5; j++){std::cout<<xpol[j]<<" "<<ypol[j]<<std::endl;}
}

/****************/
int main(int argc, char *argv[]){
  
  char *inputFile, *outputFile;
  if(argc>1)
    inputFile=argv[1];
  if(argc>2)
    outputFile=argv[2];
  xsize=340; ysize=200;
  cwidth=2000; cheight=1000;
  TPolyMarker *PM[43];
  std::string layerName[43]={"TECM9.png","TECM8.png","TECM7.png","TECM6.png","TECM5.png","TECM4.png","TECM3.png","TECM2.png","TECM1.png","TIDM3.png","TIDM2.png","TIDM1.png"," ","PIXEM2.png","PIXEM1.png","PIXEB1.png","PIXEB2.png"," ","TIDP1.png","TIDP2.png","TIDP3.png","TECP1.png","TECP2.png","TECP3.png","TECP4.png","TECP5.png","TECP6.png","TECP7.png","TECP8.png","TECP9.png","PIXB1.png","PIXB2.png","PIXB3.png","TIB1.png","TIB2.png","TIB3.png","TIB4.png","TOB1.png","TOB2.png","TOB3.png","TOB4.png","TOB5.png","TOB6.png"};
  
  
  std::ifstream *cfile;
  int cont;
  
  std::string ciccio;
 
  std::map<int,int>indice_moduli;

  cfile = new std::ifstream("tracker.dat",std::ios::in);
  
  Int_t np = 0;
  //Store data about tracker modules
  while(!cfile->eof()) {
    *cfile  >> cont >> mod[np].part  >> mod[np].subpart 
	    >> mod[np].layer  >> mod[np].ring >> mod[np].nmod
	    >> mod[np].posx >> mod[np].posy >> mod[np].posz
	    >> mod[np].length >> mod[np].width >> mod[np].thickness
	    >> mod[np].widthAtHalfLength >> mod[np].detid; 
    getline(*cfile,ciccio);
    getline(*cfile,mod[np].nome);
    indice_moduli[mod[np].detid]=np;;
    
    np++;
    
  }
  std::cout << "tracker.dat processed " << std::endl;
  MyGC = new TCanvas("MyGC","Tracker Layer",cwidth,cheight);
  for(int i=0;i<43;i++){
    PM[i] = new TPolyMarker; PM[i]->SetMarkerStyle(1); PM[i]->SetMarkerColor(1);
  }
  int modulo;
  double xp,yp,zp;
  int id;
  double * x = new double[5];
  double * y = new double[5];
  int npoints;
  std::ifstream *bfile;
  gPad->SetFillColor(10);
  gPad->Range(-10,0,400,340);
  
  
  bfile = new std::ifstream(inputFile,std::ios::in);
  
  np=0;
  saveAsSingleLayer=true;
  posrel=false;
  
  while(!bfile->eof()) {
    *bfile  >> id >> xp >> yp >> zp ;
    np++;
    
    modulo= indice_moduli[id];
    if(indice_moduli.find(id)!=indice_moduli.end() && mod[modulo].layer>0 && mod[modulo].layer<44){
     key=mod[modulo].layer*100000+mod[modulo].ring*1000+mod[modulo].nmod;
     if( (mod[modulo].nmod>100||
                    (mod[modulo].nmod<100&&!isRingStereo(key))||
                    (mod[modulo].nmod<100&&
                         (mod[modulo].layer==36||mod[modulo].layer==37||mod[modulo].layer==40||mod[modulo].layer==41||mod[modulo].layer==42||mod[modulo].layer==43)))){//use this to select stereo modules

      defwindow(mod[modulo].layer);
      if(mod[modulo].layer<31)PM[mod[modulo].layer -1]->SetNextPoint(ydpixel(yp/100.),xdpixel(xp/100.));
      else {PM[mod[modulo].layer -1]->SetNextPoint(ydpixel(4.2*phival(xp/100.,yp/100.)),xdpixel(zp/100.));
      
      }
      }
      if (np% 1000000==0)
      std::cout<<np<<" points processed" << std::endl;
    }
  }
    bfile->close();
    
    int sel_layer;

    for(int i=0;i<43;i++){
      sel_layer=i+1;
	defwindow(sel_layer);
	for (int k=0; k<16588; k++){
	  
	  if(mod[k].layer==sel_layer) {
	    computemodule(k,sel_layer,&npoints,x,y);
	    TPolyLine*  pline = new TPolyLine(npoints,x,y);
	    pline->Draw();
	  }
	  
	}
    PM[i]->Draw();
    MyGC->Update();
      if(layerName[i]!=" ")MyGC->Print(layerName[i].c_str());
      MyGC->Clear();
      delete PM[i];
     }
 return 0;

}  
  


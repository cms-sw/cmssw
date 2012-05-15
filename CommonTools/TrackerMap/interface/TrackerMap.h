#ifndef _TrackerMap_h_
#define _TrackerMap_h_
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <map>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "TColor.h"

class TmModule;
class TmApvPair;
class EventSetup;
class TmCcu;
class TmPsu;

class TrackerMap {
 public:
  //TrackerMap(){TrackerMap(" ");};   //!< default constructor
  TrackerMap(std::string s=" ",int xsize1=340,int ysize1=200);
  TrackerMap(const edm::ParameterSet & iConfig);
  TrackerMap(const edm::ParameterSet & iConfig,const edm::ESHandle<SiStripFedCabling> tkFed);
  ~TrackerMap();  //!< default destructor
  
  void build();
  void init();
  void drawModule(TmModule * mod, int key, int layer, bool total, std::ofstream * file);
  void print(bool print_total=true,float minval=0., float maxval=0.,std::string s="svgmap");
  void printall(bool print_total=true,float minval=0., float maxval=0.,std::string s="svgmap",int width=6000, int height=3200);
  void printonline();
  void printlayers(bool print_total=true,float minval=0., float maxval=0.,std::string s="layer");
  void save(bool print_total=true,float minval=0., float maxval=0.,std::string s="svgmap.svg",int width=1500, int height=800);
  void save_as_fedtrackermap(bool print_total=true,float minval=0., float maxval=0.,std::string s="fed_svgmap.svg",int width=1500, int height=800);
  void save_as_fectrackermap(bool print_total=true,float minval=0., float maxval=0.,std::string s="fec_svgmap.svg",int width=1500, int height=800);
  void save_as_psutrackermap(bool print_total=true,float minval=0., float maxval=0.,std::string s="psu_svgmap.svg",int width=1500, int height=800);
  void save_as_HVtrackermap(bool print_total=true,float minval=0., float maxval=0.,std::string s="psu_svgmap.svg",int width=1500, int height=800);
  void drawApvPair( int crate, int numfed_incrate, bool total, TmApvPair* apvPair,std::ofstream * file,bool useApvPairValue);
  void drawCcu( int crate, int numfed_incrate, bool total, TmCcu* ccu,std::ofstream * file,bool useCcuValue);
  void drawPsu(int rack,int numcrate_inrack, bool print_total, TmPsu* psu,ofstream * svgfile,bool usePsuValue);
  void drawHV2(int rack,int numcrate_inrack, bool print_total, TmPsu* psu,ofstream * svgfile,bool usePsuValue);
  void drawHV3(int rack,int numcrate_inrack, bool print_total, TmPsu* psu,ofstream * svgfile,bool usePsuValue);
  void fill_current_val(int idmod, float current_val );
  void fill(int layer , int ring, int nmod, float x );
  void fill(int idmod, float qty );
  void fillc(int idmod, int RGBcode) {fillc(idmod,(RGBcode>>16) & 0xFF , (RGBcode>>8) & 0xFF, RGBcode & 0xFF);}
  void fillc(int idmod, int red, int green, int blue);
  void fillc(int layer,int ring, int nmod, int red, int green, int blue);
  void fillc_all_blank();
  void fill_all_blank();
  void fill_current_val_fed_channel(int fedId,int fedCh, float current_val );
  void fill_fed_channel(int fedId,int fedCh, float qty );
  void fill_fed_channel(int modId, float qty );
  void fillc_fed_channel(int fedId,int fedCh, int red, int green, int blue);
  void fillc_fec_channel(int crate,int slot, int ring, int addr, int red, int green, int blue  );
  void fill_fec_channel(int crate,int slot, int ring, int addr, float qty  );
  void fill_lv_channel(int rack,int crate, int board, float qty  );
  void fillc_lv_channel(int rack,int crate, int board, int red, int green, int blue);
  void fill_hv_channel2(int rack,int crate, int board, float qty  );
  void fillc_hv_channel2(int rack,int crate, int board, int red, int green, int blue);
  void fill_hv_channel3(int rack,int crate, int board, float qty  );
  void fillc_hv_channel3(int rack,int crate, int board, int red, int green, int blue);
  int module(int fedId,int fedCh);
  void setText(int idmod , std::string s );
  void setText(int layer, int ring, int nmod , std::string s );
  void setPalette(int numpalette){palette=numpalette;} 
  void drawPalette(std::ofstream * file); 
  void showPalette(bool printflag1){printflag=printflag1;}; 
  void setTitle(std::string s){title=s;};
  void setRange(float min,float max){gminvalue=min;gmaxvalue=max;};
  std::pair<float,float>getAutomaticRange();
  void addPixel(bool addPixelfl){addPixelFlag=addPixelfl;};
  void reset();
  void load(std::string s="tmap.svg"); 
  int getxsize(){return xsize;};
  int getysize(){return ysize;};
  int getcolor(float value, int palette);
  std::ifstream * findfile(std::string filename);
  int getNumMod(){return number_modules;};
  std::vector<TColor*> vc; 
  typedef std::map<const int  , TmModule *> SmoduleMap;
  SmoduleMap smoduleMap;
  typedef std::map<const int  , TmModule *> ImoduleMap;
  ImoduleMap imoduleMap;
  typedef std::map<const int  , TmApvPair*> SvgApvPair;
  SvgApvPair apvMap;
  typedef std::multimap<const int  , TmApvPair*> ModApvPair;
   ModApvPair apvModuleMap;
  typedef std::map<const int  , int>  SvgFed;
  SvgFed fedMap;
  typedef std::map<const int  , TmCcu*> MapCcu;
  MapCcu  ccuMap;
  typedef std::multimap<TmCcu*  , TmModule*> FecModule;
  FecModule fecModuleMap;
  typedef std::map<const int  , TmPsu*> MapPsu;
  MapPsu  psuMap;
  typedef std::multimap<TmPsu*  , TmModule*> PsuModule;
  PsuModule psuModuleMap;
  int palette;
  bool printflag;
  bool saveWebInterface;
  bool saveGeoTrackerMap;
  bool enableFedProcessing;
  bool enableFecProcessing;
  bool enableLVProcessing;
  bool enableHVProcessing;
  int ndet; //number of detectors 
  int npart; //number of detectors parts 
  std::string title;
   std::string jsfilename,infilename;
  std::string jsPath;
   bool psetAvailable;
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
  
  int find_layer(int ix, int iy)
    {
      int add;
      int layer=0;
      if(iy <= xsize){//endcap+z
        add = 15;
	layer = ix/ysize;
	layer = layer+add+1;
      }
      if(iy > xsize && iy< 3*xsize){//barrel
	add=30;
	if(ix < 2*ysize){
	  layer=1;
	}else {
	  layer = ix/(2*ysize);
	  if(iy < 2*xsize)layer=layer*2+1; else layer=layer*2;
     	}
	layer = layer+add;
      }
      if(iy >= 3*xsize){	//endcap-z
	layer = ix/ysize;
	layer = 15-layer;
      }
      return layer;  
    }

  int getlayerCount(int subdet, int partdet){
    int ncomponent=0;
    if(subdet == 1){ //1=pixel
      if(partdet == 1 || partdet == 3){ //1-3=encap
	ncomponent = 3;
      }
      else { ncomponent = 3; } //barrel
    }
    if(subdet== 2){ //2=inner silicon
      if(partdet == 1 || partdet == 3){ //1-3=encap
	ncomponent = 3;
      }
      else { ncomponent = 4; } //barrel
    }
    if(subdet== 3){ //3=outer silicon
      if(partdet == 1 || partdet == 3){ //1-3=encap
	ncomponent = 9;
      }
      else { ncomponent = 6; } //barrel
    }
    return(ncomponent);
  }   
  double  xdpixel(double x){
    double res;
    if(saveAsSingleLayer)res= ((x-xmin)/(xmax-xmin)*xsize);
    else res= ((x-xmin)/(xmax-xmin)*xsize)+ix;
    return res;
  }
  double  ydpixel(double y){
    double res=0;
    double y1;
    y1 = (y-ymin)/(ymax-ymin);
    if(nlay>30)
       {
        if(nlay <34) res= 2*ysize - (y1*2*ysize);
        if(nlay==34) res= 2.4*ysize - (y1*2.4*ysize);
        if(nlay>34) res= 2.5*ysize - (y1*2.5*ysize);  
        }
    else res= xsize - (y1*xsize);
    if(!saveAsSingleLayer) res=res+iy;
    return res;
  }
  double  xdpixelc(double x){
    double res;
    if(saveAsSingleLayer)res= ((x-xmin)/(xmax-xmin)*xsize);
    else res= ((x-xmin)/(xmax-xmin)*xsize)+ix;
    return res;
  }
  double  ydpixelc(double y){
    double res;
    double y1;
    y1 = (y-ymin)/(ymax-ymin);
     if(saveAsSingleLayer)res= 2*ysize - (y1*2*ysize);
     else res= 2*ysize - (y1*2*ysize)+iy;
    return res;
  }
  double  xdpixelfec(double x){
    double res;
    if(saveAsSingleLayer)res= ((x-xmin)/(xmax-xmin)*xsize);
    else res= ((x-xmin)/(xmax-xmin)*xsize)+ix;
    return res;
  }
  double  ydpixelfec(double y){
    double res;
    double y1;
    y1 = (y-ymin)/(ymax-ymin);
     if(saveAsSingleLayer)res= 2*ysize - (y1*2*ysize);
     else res= 2*ysize - (y1*2*ysize)+iy;
    return res;
  }
  double  xdpixelpsu(double x){
    double res;
    if(saveAsSingleLayer)res= ((x-xmin)/(xmax-xmin)*xsize);
    else res= ((x-xmin)/(xmax-xmin)*xsize)+ix;
    return res;
  }
   double  ydpixelpsu(double y){
    double res;
    double y1;
    y1 = (y-ymin)/(ymax-ymin);
     if(saveAsSingleLayer)res= 2*ysize - (y1*2*ysize);
     else res= 2*ysize - (y1*2*ysize)+iy;
    return res;
  }

   void defcwindow(int num_crate){
    ncrate = num_crate;
    int xoffset=xsize/3;
    int yoffset=ysize;
    xmin=-1.;xmax=63.;  ymin = -1.; ymax=37.;
    if((ncrate%3)==2)ix = xoffset+xsize*4/3;
    if((ncrate%3)==1)ix = xoffset+2*xsize*4/3;
    if((ncrate%3)==0)ix = xoffset;
    iy = yoffset+((ncrate-1)/3)*ysize*2;
  } 
   void deffecwindow(int num_crate){
    ncrate = num_crate;
    int xoffset=xsize/3;
    int yoffset=2*ysize;
    xmin=-1.;xmax=37.;  ymin = -10.; ymax=40.;
    if(ncrate==1||ncrate==3)ix = xoffset+xsize*2;
    if(ncrate==2||ncrate==4)ix = xoffset;
    iy = yoffset+((ncrate-1)/2)*ysize*4;
  }
   void defpsuwindow(int num_rack){
    nrack = num_rack;
    int xoffset=xsize/5;
    int yoffset=ysize;
    xmin=-1.;xmax=63.;  ymin = -1.; ymax=37.;

    if((nrack%5)==1)ix = xoffset+4*int(xsize/1.5);
    if((nrack%5)==2)ix = xoffset+3*int(xsize/1.5);
    if((nrack%5)==3)ix = xoffset+2*int(xsize/1.5);
    if((nrack%5)==4)ix = xoffset+int(xsize/1.5);
    if((nrack%5)==0)ix = xoffset;

    iy = yoffset+((nrack-1)/5)*ysize*2;

    }


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
      ix=0;
      if(nlay==15||nlay==14)iy=(15-nlay)*2*ysize; else 
          {if(nlay>9&&nlay<13)iy=4*ysize-(int)(ysize/2.)+(12-nlay)*(int)(ysize/1.50);else iy=6*ysize+(9-nlay)*(int)(ysize*1.3);}}
  if(nlay>15&&nlay<31){
    ix=3*xsize;
     if(nlay==16||nlay==17)iy=(nlay-16)*2*ysize; else 
          {if(nlay>18&&nlay<22)iy=4*ysize-(int)(ysize/2.)+(nlay-19)*(int)(ysize/1.50);else iy=6*ysize+(nlay-22)*(int)(ysize*1.3);}}
  if(nlay>30){
    if(nlay==31){ix=(int)(1.5*xsize);iy=0;}
    if(nlay==32){int il=(nlay-30)/2;ix=xsize;iy=il*2*ysize;}
    if(nlay==33){int il=(nlay-30)/2;ix=2*xsize;iy=il*2*ysize;}
    if(nlay==34){int il=(nlay-30)/2;ix=xsize;iy=il*(int)(2.57*ysize);}
    if(nlay>34 && nlay%2==0){int il=(nlay-30)/2;ix=xsize;iy=il*(int)(2.5*ysize);}
    if(nlay>34 && nlay%2!=0){int il=(nlay-30)/2;ix=2*xsize;iy=il*(int)(2.5*ysize);}
  }
 }
  
  int getringCount(int subdet, int partdet, int layer){
    int ncomponent=0;
    if(subdet== 1){ //1=pixel
      if(partdet== 1 || partdet== 3){ //end-cap
	ncomponent = 7;
      }
      else{ncomponent = 8;} //barrel
    }	
    if(subdet== 2){ //inner-silicon
      if(partdet== 1 || partdet== 3){ //end-cap
	ncomponent = 3;
      }
      else{ncomponent = 12;} //barrel
    }
    if(subdet== 3){ //outer-silicon
      if(partdet== 1){ //end-cap-z
	if (layer== 1) ncomponent = 4;
	if (layer== 2 || layer== 3) ncomponent = 5;
	if (layer== 4 || layer== 5 || layer== 6) ncomponent = 6;
	if (layer== 7 || layer== 8 || layer== 9) ncomponent = 7;
      }
      if(partdet== 3){ //endcap+z
	if (layer== 9) ncomponent = 4;
	if (layer== 8 || layer== 7) ncomponent = 5;
	if (layer== 6 || layer== 5 || layer== 4) ncomponent = 6;
	if (layer== 3 || layer== 2 || layer== 1) ncomponent = 7;
      }
      if(partdet== 2){ //barrel
	ncomponent = 12;
      }
    }
    return(ncomponent);
  }
  int getmoduleCount(int subdet, int partdet, int layer, int ring){
    int ncomponent=0;
    int spicchif[] ={24,24,40,56,40,56,80};
    int spicchib[] ={20,32,44,30,38,46,56,42,48,54,60,66,74};
    int numero_layer = 0;
    
    if(partdet == 2){ //barrel
      numero_layer = layer-1;
      if(subdet== 2){ //inner
	numero_layer = numero_layer+3;
      }
      if(subdet == 3){ //outer
	numero_layer = numero_layer+7;
      }
      ncomponent = spicchib[numero_layer];
    }
    if(partdet!= 2){ //endcap
      if(subdet== 1)ncomponent=24;//pixel
      else
	ncomponent = spicchif[ring-1];
    }
    return(ncomponent);
  }
  static int layerno(int subdet,int leftright,int layer){
    if(subdet==6&&leftright==1)return(10-layer);
    if(subdet==6&&leftright==2)return(layer+21);
    if(subdet==4&&leftright==1)return(4-layer+9);
    if(subdet==4&&leftright==2)return(layer+18);
    if(subdet==2&&leftright==1)return(4-layer+12);
    if(subdet==2&&leftright==2)return(layer+15);
    if(subdet==1)return(layer+30);
    if(subdet==3)return(layer+33);
    if(subdet==5)return(layer+37);
  }
  
  static bool isRingStereo(int key){
    int layer=key/100000;
    int ring = key - layer*100000;
    ring = ring/1000;
    if(layer==34 || layer==35 || layer==38 || layer==39) return true;
    if(layer<13 || (layer>18&&layer<31))
      if(ring==1 || ring==2 || ring==5)return true;
    return false;
  }
  int nlayer(int det,int part,int lay){
    if(det==3 && part==1) return lay;
    if(det==2 && part==1) return lay+9;
    if(det==1 && part==1) return lay+12;
    if(det==1 && part==3) return lay+15;
    if(det==2 && part==3) return lay+18;
    if(det==3 && part==3) return lay+21;
    if(det==1 && part==2) return lay+30;
    if(det==2 && part==2) return lay+33;
    if(det==3 && part==2) return lay+37;
    return -1; 
  }
  
  std::string layername(int layer){
    std::string s= " ";
    std::ostringstream ons;
    
    if(layer < 10) ons << "TEC -z Layer " << layer;
    if(layer < 13 && layer > 9) ons << "TID -z Layer " << layer-9;
    if(layer < 16 && layer > 12) ons << "FPIX -z Layer " << layer-12;
    if(layer < 19 && layer > 15) ons << "FPIX +z Layer " << layer-15;
    if(layer < 22 && layer > 18) ons << "TID +z Layer " << layer-18;
    if(layer < 31 && layer > 21) ons << "TEC +z Layer " << layer-21;
    if(layer < 34 && layer > 30) ons << "TPB Layer " << layer-30;
    if(layer < 38 && layer > 33) ons << "TIB Layer " << layer-33;
    if(layer > 37) ons << "TOB Layer " << layer-37;
    s = ons.str(); 
    return s;  
  }
  int ntotRing[43];
  int firstRing[43];
  
 protected:
  int nlay;
  int ncrate;
  int nrack;
  int ncrates;
  int nfeccrates;
  int npsuracks;
  double xmin,xmax,ymin,ymax;
  int xsize,ysize,ix,iy;
  bool posrel;
  bool firstcall;
  std::ofstream * svgfile;
  std::ofstream * savefile;
  std::ifstream * jsfile;
  std::ifstream * inputfile;
  std::ifstream * ccufile;
  float gminvalue,gmaxvalue;
  float minvalue,maxvalue;
  int number_modules;
  bool temporary_file; 
  
 private:
  
  float oldz;
  bool saveAsSingleLayer;
  bool addPixelFlag;
};
#endif


#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <map>

class TmModule;
class EventSetup;

using namespace std;

class TrackerMap {
 public:
  //TrackerMap(){TrackerMap(" ");};   //!< default constructor
  TrackerMap(std::string s=" ",int xsize1=340,int ysize1=200);
  ~TrackerMap();  //!< default destructor
  
  void build();
  void drawModule(TmModule * mod, int key, int layer, bool total, std::ofstream * file);
  void print(bool print_total=true,float minval=0., float maxval=0.,std::string s="svgmap");
  void save(bool print_total=true,float minval=0., float maxval=0.,std::string s="svgmap.svg",int width=1500, int height=800);
  void fill_current_val(int idmod, float current_val );
  void fill(int layer , int ring, int nmod, float x );
  void fill(int idmod, float qty );
  void fillc(int idmod, int RGBcode) {fillc(idmod,(RGBcode>>16) & 0xFF , (RGBcode>>8) & 0xFF, RGBcode & 0xFF);}
  void fillc(int idmod, int red, int green, int blue);
  void fillc(int layer,int ring, int nmod, int red, int green, int blue);
  void setText(int idmod , std::string s );
  void setText(int layer, int ring, int nmod , string s );
  void setPalette(int numpalette){palette=numpalette;} 
  void drawPalette(std::ofstream * file); 
  void showPalette(bool printflag1){printflag=printflag1;}; 
  int getxsize(){return xsize;};
  int getysize(){return ysize;};
  int getNumMod(){return number_modules;};
  typedef std::map<const int  , TmModule *> SmoduleMap;
  SmoduleMap smoduleMap;
  typedef std::map<const int  , TmModule *> ImoduleMap;
  ImoduleMap imoduleMap;
  int palette;
  bool printflag;
  int ndet; //number of detectors 
  int npart; //number of detectors parts 
  string title;
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
    res= ((x-xmin)/(xmax-xmin)*xsize)+ix;
    return res;
  }
  double  ydpixel(double y){
    double res;
    double y1;
    y1 = (y-ymin)/(ymax-ymin);
    if(nlay>30)   res= 2*ysize - (y1*2*ysize)+iy;
    else res= ysize - (y1*ysize)+iy;
    return res;
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
      ix=0;iy=(15-nlay)*ysize;}
  if(nlay>15&&nlay<31){
    ix=3*xsize;iy=(nlay-16)*ysize;}
  if(nlay>30){
    if(nlay==31){ix=(int)(1.5*xsize);iy=0;}
    if(nlay>31 && nlay%2==0){int il=(nlay-30)/2;ix=xsize;iy=il*2*ysize;}
    if(nlay>31 && nlay%2!=0){int il=(nlay-30)/2;ix=2*xsize;iy=il*2*ysize;}
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
  
  string layername(int layer){
    string s= " ";
    ostringstream ons;
    
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
  double xmin,xmax,ymin,ymax;
  int xsize,ysize,ix,iy;
  bool posrel;
  bool firstcall;
  std::ofstream * svgfile;
  std::ofstream * savefile;
  std::ifstream * jsfile;
  float minvalue,maxvalue;
  int number_modules;
  
 private:
  
  float oldz;
};



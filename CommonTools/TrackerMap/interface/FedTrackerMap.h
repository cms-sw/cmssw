#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
class TmApvPair;
class TmModule;

using namespace std;

class FedTrackerMap : public TrackerMap {
 public:
  FedTrackerMap(const edm::ESHandle<SiStripFedCabling> tkFed);   //!< default constructor
  ~FedTrackerMap();  //!< default destructor
  
  void drawApvPair( int layer, int numfed_inlayer, bool total, TmApvPair* apvPair);
  void print(bool print_total=true,float minval=0., float maxval=0.);
  void fill_current_val(int fedId,int fedCh, float current_val );
  void fill(int fedId,int fedCh, float qty );
  void fillc(int fedId,int fedCh, int red, int green, int blue);
  void setText(int fedId,int fedCh , string s );
  void defwindow(int num_lay){
    nlay = num_lay;
    xmin=-1.;xmax=50.;  ymin = -1.; ymax=26.;
    if(nlay>30)
      {xmin=-1.;xmax=50.;ymin=-1.;   ymax=51.;}
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
  
 private:
  
  string title;
  ofstream * svgfile;
  ifstream * jsfile;
  float minvalue,maxvalue;
};



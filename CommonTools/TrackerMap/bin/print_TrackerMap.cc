#include <iostream>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

void printTrackerMap(const std::string filename, const std::string title, const std::string outfile, const int size, const std::string logscale, std::string withpixel, const float min, const float max);

int main(int argc, char *argv[]) {

  if(argc>=4) {

    int size=4500;
    float min = 10.;
    float max = -1.;
    std::string logscale = "False";
    std::string withpixel = "False";
    
    char* filename =argv[1];
    char* title = argv[2];
    char* outfile = argv[3];
    std::cout << "ready to use file " << filename << " to create the map " << title << " and save it in " <<outfile << std::endl;

    if(argc>=5) {
      size = atoi(argv[4]);
    }
    if(argc>=6) {
      logscale = argv[5];
      withpixel = argv[6];
    }
    if(argc>=9) {
      min = atof(argv[7]);
      max = atof(argv[8]);
    }

    printTrackerMap(filename,title,outfile,size,logscale,withpixel,min,max);

  }
  else { std::cout << "Wrong number of arguments: " << argc << std::endl; return -1; }

  return 0;
}

void printTrackerMap(const std::string filename, const std::string title, const std::string outfile, 
		     const int size, const std::string logscale, std::string withpixel, const float min, const float max) {

  
  edm::ParameterSet pset;
  
  if(logscale=="True") pset.addUntrackedParameter<bool>("logScale",true);
  if(logscale=="False") pset.addUntrackedParameter<bool>("logScale",false);

  TrackerMap themap(pset);
  themap.setTitle(title); // title as input parameter
  double ratio=2400./4500.;
  if(withpixel=="True") {themap.addPixel(true); themap.onlyPixel(false); ratio=8./19.;}
  if(withpixel=="Only") {themap.addPixel(false); themap.onlyPixel(true); ratio=16./9.;}
  if(withpixel=="False") {themap.addPixel(false); themap.onlyPixel(false); ratio=8./15.;}

  std::ifstream input(filename);

  unsigned int detid;
  float val;

  while( input >> detid >> val) {
    themap.fill_current_val(detid,val);
  }

  std::cout << "preparing a map with size " << size << "x" << int(size*ratio) << std::endl;
 
  themap.save(true,min,max,outfile,size,int(size*ratio)); // name and size as input parameters



}

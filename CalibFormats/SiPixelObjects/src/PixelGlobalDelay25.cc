//
// This class stores the information about the global delay25 delay settings
// For the time being, the affected delay25 channels are SDA, SCL, and TRG
// (as implemented in PixelTKFECSupervisor)
//

#include "CalibFormats/SiPixelObjects/interface/PixelGlobalDelay25.h"
#include <fstream>
#include <map>
#include <assert.h>
#include <math.h>


using namespace pos;
using namespace std;


PixelGlobalDelay25::PixelGlobalDelay25(std::string filename):
    PixelConfigBase(" "," "," "){

    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << "Could not open:"<<filename<<std::endl;
	assert(0);
    }
    else {
	std::cout << "Opened:"<<filename<<std::endl;
    }

    in >> std::hex >> delay_ >> std::dec;
    std::cout << "PixelGlobalDelay25:: read global delay 0x" << std::hex << delay_ << std::dec << endl;  

    in.close();

    if (delay_>=50) {
      std::cout << "PixelGlobalDelay25:: global delay is out of range (>= 1 Tclk)."<< std::endl;
      std::cout << "PixelGlobalDelay25:: will not apply any global delays."<<std::endl;
      std::cout << "PixelGlobalDelay25:: increase the delays in the TPLL if needed."<<std::endl;
      delay_=0;
    }
}

 
PixelGlobalDelay25::~PixelGlobalDelay25() {}


unsigned int PixelGlobalDelay25::getDelay(unsigned int offset) const{
  unsigned int ret=offset+delay_;
  if (ret > 127) {
    std::cout<<"PixelGlobalDelay25:: the required Delay25 delay "<<ret<<" is out of range."<<endl;
    std::cout<<"PixelGlobalDelay25:: this can happen if the operating point is chosen to be"<<endl;
    std::cout<<"PixelGlobalDelay25:: over 78; i.e. more than 7 ns delay."<<endl;    
    std::cout<<"PixelGlobalDelay25:: we will keep the current delay setting..."<<endl;
    ret=offset;
  }

  std::cout<<"PixelGlobalDelay25::getDelay("<<offset<<") returns "<<ret<<endl;
  return ret;
}


unsigned int PixelGlobalDelay25::getTTCrxDelay(unsigned int offset) const{
  // Computes the TTCrx delay settting required to compensate for the global Delay25 shift.
  //
  // 'offset' is the current register setting in the TTCrx.
  //
  // The unit of delay_ is 0.499 ns (Delay25 granularity) that needs to be converted
  // to the units of the TTCrx delay generator 103.96 ps

  unsigned int K=(offset/16*16+offset%16*15+30)%240;
  K+=floor((delay_*0.499)/0.1039583 + 0.5); // add max 235

  unsigned int ret;
  if (K>239) {
    std::cout<<"PixelGlobalDelay25:: the required TTCrx fine delay "<<K<<" is out of range."<<endl;
    std::cout<<"PixelGlobalDelay25:: this can happen if the register was initialized to 0"<<endl;
    std::cout<<"PixelGlobalDelay25:: (i.e. delay of 3.1 ns) and the required delay is >21.7 ns."<<endl;    
    std::cout<<"PixelGlobalDelay25:: we will keep the current delay setting..."<<endl;
    ret=offset;
  }else{
    unsigned int n=K%15;
    unsigned int m=((K/15)-n+14)%16;
    ret=16*n+m;
  }
  
  std::cout<<"PixelGlobalDelay25::getTTCrxDelay("<<offset<<") returns "<<ret<<endl;
  return ret;
  //return offset;
}


void PixelGlobalDelay25::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  string filename=dir+"globaldelay25.dat";

  ofstream out(filename.c_str());
  if(!out.good()){
    cout << "Could not open file:"<<filename<<endl;
    assert(0);
  }

  out << "0x" << hex << delay_ << dec << endl;

  out.close();
}

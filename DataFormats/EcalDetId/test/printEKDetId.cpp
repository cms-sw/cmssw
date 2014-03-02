// prints EKDetId mappings for humans to check over

#include <iostream>
#include <string>
#include <stdexcept>
#include <iomanip>

#include "DataFormats/EcalDetId/interface/EKDetId.h"

std::ostream& pcenter(unsigned w, std::string s){
  int pad = ((int)w - (int)s.size()) / 2;
  //  if(pad<0) pad = 0;
  for(int i = 0; i < pad; ++i) std::cout << " ";
  std::cout << s;
  for(int i = pad + s.size(); i < (int)w; ++i) std::cout << " ";
  return std::cout;
}

int main(int argc, char* argv[]) {
  const int colsize = 8;
#define COL std::cout << std::setw(colsize)
  
  std::cout << std::right;
  pcenter(5*(colsize+1)-1, "input") << "|";
  pcenter(12*(colsize+1), "detid") << "|";
  pcenter(4*(colsize+1), "detid->(ix,iy,fib,ro,iz)->detid") << "\n";
  
  //input
  COL <<  "ism" << " ";
  COL <<  "im" << " ";
  COL <<  "fib" << " ";
  COL <<  "ro" << " ";
  COL <<  "iz" << "|";
  
  //detId
  COL <<  "ix" << " ";
  COL <<  "iy" << " ";
  COL <<  "fib" << " ";
  COL <<  "ro" << " ";
  COL <<  "zside" << " ";
  COL <<  "iquad" << " ";
  COL <<  "Z" << " ";
  COL <<  "hash_ind" << " ";
  COL <<  "hash_chk" << " ";
  COL <<  "dense_ind" <<  " ";
  COL <<  "ism" << " ";
  COL <<  "im" << "|";

  //detid->isc,ic->detid
  COL <<  "ix" << " ";
  COL <<  "iy" << " ";
  COL <<  "iz" << " ";
  COL <<  "iscic_chk" << "\n";


  try {
    for(int iz = -1 ; iz <= 1; iz += 2){
      COL << "========== " << (iz>0 ? "EK+" : "EK-" ) << " ========== \n";
      for(int ism = 1; ism <= 1000; ++ism) {
	for(int im = 1; im <= 30; ++im) {
	  for(int fib = 0; fib <= 1; ++fib) {
	    for(int ro = 0; ro <= 1; ++ro) {
	      if(!EKDetId::validDetId(ism,im,fib,ro,iz)) continue;
	  
	      //input
	      COL << ism << " ";
	      COL << im << " ";
	      COL << fib << " ";
	      COL << ro << " ";
	      COL << iz << " ";
	      //detid
	      EKDetId id(ism,im,fib,ro,iz,EKDetId::SCMODULEMODE) ;
	      COL << id.ism() << (ism != id.ism() ? "!!!" : "") << " ";
	      COL << id.imod() << (im != id.imod() ? "!!!" : "") << " ";
	      COL << id.fiber() << (fib != id.fiber() ? "!!!" : "") << " ";
	      COL << id.readout() << (ro != id.readout() ? "!!!" : "") << " ";
	      COL << id.zside() << (iz != id.zside() ? "!!!" : "") << " ";
	      COL << id.iquadrant() << " ";
	      COL << (id.positiveZ() ? "z+" : "z-") <<  (id.positiveZ() != (iz > 0) ? "!!!" : "" ) << " ";
	  
	      //hashed index
	      int ih = id.hashedIndex() ;
	      COL << ih << " ";
	      if(!EKDetId::validHashIndex(ih) || EKDetId::unhashIndex(ih)!=id || EKDetId::unhashIndex(ih).rawId() != id.rawId()){
		COL << "ERR!!!" << " ";
	      } else{
		COL << "OK" << " ";
	      }
	  
	      COL << id.denseIndex() << (id.denseIndex() != (uint32_t)ih ? "!!!" : "") << " ";
	  
	      //IXY
	      const int ix = id.ix();
	      COL << ix << " ";
	      const int iy = id.iy();
	      COL << iy << " ";
	      EKDetId id1(ix, iy, fib, ro, iz);
	      COL << id1.ix()  << (id1.ix()!=ix ? "!!!" : "") << " ";
	      COL << id1.iy()  << (id1.iy()!=iy ? "!!!" : "") << " ";
	      COL << id1.zside()  << (id1.zside()!=iz ? "!!!" : "") << " ";			    
	      if(id!=id1 || id.rawId()!=id1.rawId()){
		COL << "ERR!!!" << " ";
	      } else{
		COL << "OK" << " ";
	      }
	      COL << "\n";
	    } //next ro
	  } // next fiber
	} //next im
      } //next ism
    } //next iz
  } catch (std::exception &e) {
    std::cerr << e.what();
  }
}

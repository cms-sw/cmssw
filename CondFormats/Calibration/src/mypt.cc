#include "CondFormats/Calibration/interface/mypt.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
//#include <iostream>
//typedef boost::minstd_rand base_generator_type;
void mypt::fill(){
  //base_generator_type rng(42u);
  //boost::uniform_int<int> uni_dist(0,200);
  //boost::variate_generator<base_generator_type&, boost::uniform_int<> > uni(rng, uni_dist); 
  for(size_t i=0;i<2097;++i){
    std::cout<<"i "<<i<<" ";
    pt[i]=(unsigned short)(i);
  }
}

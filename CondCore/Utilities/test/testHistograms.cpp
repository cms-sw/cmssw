#include<iostream>
#include<sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"

int main() {

  std::cout <<"## Histo1D"<<std::endl;
  cond::payloadInspector::Histogram1D<float> histo1( "histo1","x",7,5.,15.);
  for( size_t i=0; i<20; i++ ) histo1.fillWithValue( (float)i );   
  std::cout << histo1.serializeData()<<std::endl;
  std::cout <<"## Histo2D"<<std::endl;
  cond::payloadInspector::Histogram2D<float> histo2( "histo2","x",7,5.,15.,"y",10,1,20);
  for( size_t i=0; i<20; i++ ){
    for( size_t j=0; j<25; j++ ) histo2.fillWithValue( (float)i,j );  
  } 
  std::cout << histo2.serializeData()<<std::endl;
}

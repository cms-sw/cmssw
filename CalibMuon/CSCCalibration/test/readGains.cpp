#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

using namespace std;

int main(){
  float gainSlope, gainNonLin,normal;
  int index;
  int nrlines=0;

  std::vector<int>   index_id;
  std::vector<float> Slope;
  std::vector<float> NonLinearity;
  std::vector<float> Normalization;
  
  std::ifstream dbdata; 
  dbdata.open("gainSummary2008_09_02.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: gainSummary2008_09_02.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while (!dbdata.eof() ) { 
    dbdata >> index >> gainSlope >> gainNonLin >> normal ; 
    index_id.push_back(index);
    Slope.push_back(gainSlope);
    NonLinearity.push_back(gainNonLin);
    Normalization.push_back(normal);
    nrlines++;
  }
  dbdata.close();
  std::ofstream myGainsFile("goodGains2008_09_02.dat",std::ios::out);
 
  for(int i=0; i<nrlines-1;++i){
    if (Slope[i]!=0.0 && Slope[i]>6.0 && Slope[i]<11.0 && NonLinearity[i]<50.0){
      myGainsFile<<index_id[i]<<"  "<<Slope[i]<<"  "<<NonLinearity[i]<<"  "<<Normalization[i]<<std::endl;
      if (Slope[i]<6){
	std::cout<<"Index "<<index_id[i]<<"  "<<Slope[i]<<std::endl;
      }
    }
  }
}

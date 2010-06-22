#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

using namespace std;

int main(){
  float gainSlope, gainIntercept,chi2;
  int index,flag,flag1;
  int nrlines=0;

  std::vector<int>   index_id;
  std::vector<float> Slope;
  std::vector<float> Intercept;
  std::vector<float> gainChi2;
  
  std::ifstream dbdata; 
  dbdata.open("/nfshome0/boeriu/cal_data/merged_data/gainSummary2010_03_18_run131352.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: gainSummary2010_03_18_run131352.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while (!dbdata.eof() ) { 
    dbdata >> index >> gainSlope >> gainIntercept >>chi2 >>flag >>flag1; 
    index_id.push_back(index);
    Slope.push_back(gainSlope);
    Intercept.push_back(gainIntercept);
    gainChi2.push_back(chi2);
    nrlines++;
  }
  dbdata.close();
  std::ofstream myGainsFile("goodGains2010_03_18_run131352.dat",std::ios::out);
 
  for(int i=0; i<nrlines-1;++i){
    if (Slope[i]!=0.0 && Slope[i]>6.0 && Slope[i]<11.0){
      myGainsFile<<index_id[i]<<"  "<<Slope[i]<<"  "<<Intercept[i]<<"  "<<gainChi2[i]<<std::endl;
      if (Slope[i]<6){
	std::cout<<"Index "<<index_id[i]<<"  "<<Slope[i]<<std::endl;
      }
      if (flag==1 || flag1==1){
	std::cout<<"Flag not 0: "<<index_id[i]<<" " <<flag<<"  "<<flag1<<std::endl;
      }
    }
  }
}

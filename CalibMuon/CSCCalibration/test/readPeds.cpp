#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

using namespace std;

int main(){
  float peds,chi2;
  int index,flag,flag1;
  int nrlines=0;

  std::vector<int>   index_id;
  std::vector<float> Ped;
  std::vector<float> pedChi2;
  
  std::ifstream dbdata; 
  dbdata.open("FileName",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: FileName -> no such file!"<< std::endl;
    exit(1);
  }

  while (!dbdata.eof() ) { 
    dbdata >> index >> peds >>chi2 >>flag >>flag1; 
    index_id.push_back(index);
    Ped.push_back(peds);
    pedChi2.push_back(chi2);
    nrlines++;
  }
  dbdata.close();
  std::ofstream myPedsFile("GoodVals_FileName",std::ios::out);
 
  for(int i=0; i<nrlines-1;++i){
    if (Ped[i]>400.0 && Ped[i]<1000.0){
      myPedsFile<<index_id[i]<<"  "<<Ped[i]<<"  "<<pedChi2[i]<<std::endl;
      if (flag==1 || flag1==1){
  std::cout<<"Flag not 0: "<<index_id[i]<<" " <<flag<<"  "<<flag1<<std::endl;
      }
    }
  }
}

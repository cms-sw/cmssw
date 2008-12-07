#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>


using namespace std;

int main(){
  float leftslope,leftint,rightslope,rightint;
  int index;
  int nrlines1=0;

  std::vector<int> index_id;
  std::vector<float> leftSlope;
  std::vector<float> leftInt;
  std::vector<float> leftChi2;
  std::vector<float> rightSlope;
  std::vector<float> rightInt;

  std::ifstream newdata; 
  newdata.open("xtalkSummary2008_10_28.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: xtalkSummary2008_10_28.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while (!newdata.eof() ) { 
    newdata >> index >> leftslope >> leftint >> rightslope >> rightint ; 
    index_id.push_back(index);
    leftSlope.push_back(leftslope);
    leftInt.push_back(leftint);
    rightSlope.push_back(rightslope);
    rightInt.push_back(rightint);
    nrlines1++;
  }
  newdata.close();

  std::ofstream myXtalkFile("goodXtalk.dat",std::ios::out);
  for(int i=0; i<nrlines1-1;++i){
    leftChi2[i]=0.0;
     if (rightSlope[i]<-0.001 && rightInt[i]<0.1 && leftSlope[i]<-0.001 && leftInt[i]<0.1){
       myXtalkFile<<index_id[i]<<"  "<<leftSlope[i]<<"  "<<leftInt[i]<<"   "<<leftChi2[i]<<"  "<<rightSlope[i]<<"   "<<rightInt[i]<<"  "<<leftChi2[i]<<std::endl;
       //std::cout<<"Warning! Xtalk out of range!!"<<index_id1[i]<<"  "<<leftSlope[i]<<std::endl;
    }
  }
}

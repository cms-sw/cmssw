#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>


using namespace std;

int main(){
  float leftslope,leftint,rightslope,rightint;
  int index,extra1,extra2;
  int nrlines1=0;

  std::vector<int> index_id;
  std::vector<float> leftSlope;
  std::vector<float> leftInt;
  std::vector<float> leftChi2;
  std::vector<float> rightSlope;
  std::vector<float> rightInt;
 
  std::ifstream newdata; 
  newdata.open("/nfshome0/boeriu/cal_data/merged_data/xtalkSummary2010_03_18_run131357.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: xtalkSummary2010_03_18_run131357.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while (!newdata.eof() ) { 
    newdata >> index >> leftslope >> leftint >> rightslope >> rightint >> extra1 >>extra2;
    index_id.push_back(index);
    leftSlope.push_back(leftslope);
    leftInt.push_back(leftint);
    rightSlope.push_back(rightslope);
    rightInt.push_back(rightint);
    nrlines1++;
  }
  newdata.close();

  std::ofstream myXtalkFile("goodXtalk2010_03_18_run131357.dat",std::ios::out);
  for(int i=0; i<nrlines1-1;++i){
    //    if (rightSlope[i]<-0.001 && rightSlope[i]!=-999 && rightInt[i]<0.1 && leftSlope[i]<-0.001 && leftInt[i]<0.1 ){
      if (rightSlope[i]!=-999 && leftSlope[i]!=-999 &&  leftInt[i]!=-999 & rightInt[i]!=-999 && extra1 !=1){
	myXtalkFile<<index_id[i]<<"  "<<leftSlope[i]<<"  "<<leftInt[i]<<"  "<<rightSlope[i]<<"   "<<rightInt[i]<<std::endl;
	//std::cout<<"Warning! Xtalk out of range!!"<<index_id1[i]<<"  "<<leftSlope[i]<<std::endl;
	//}
    }
      if (extra1==1 || extra2==1){
	std::cout<<"Flag not 0: "<<index_id[i]<<" " <<extra1<<"  "<<extra2<<std::endl;
      }
  }
}

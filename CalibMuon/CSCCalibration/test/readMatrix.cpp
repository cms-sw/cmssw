#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

using namespace std;

int main(){
  float elem33,elem34,elem44,elem35,elem45,elem55,elem46,elem56,elem66,elem57,elem67,elem77;
  int index;
  int nrlines=0;

  std::vector<int>   index_id;
  std::vector<float> Elem33;
  std::vector<float> Elem34;
  std::vector<float> Elem44;
  std::vector<float> Elem35;
  std::vector<float> Elem45;
  std::vector<float> Elem55;
  std::vector<float> Elem46;
  std::vector<float> Elem56;
  std::vector<float> Elem66;
  std::vector<float> Elem57;
  std::vector<float> Elem67;
  std::vector<float> Elem77;
   
  std::ifstream dbdata; 
  dbdata.open("matrixSummary2008_09_02_fixed.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: matrixSummary2008_09_02_fixed.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while (!dbdata.eof() ) { 
    dbdata >> index >>elem33>>elem34>>elem44>>elem35>>elem45>>elem55>>elem46>>elem56>>elem66>>elem57>>elem67>>elem77 ; 
    index_id.push_back(index);
    Elem33.push_back(elem33);
    Elem34.push_back(elem34);
    Elem44.push_back(elem44);
    Elem35.push_back(elem35);
    Elem45.push_back(elem45);
    Elem55.push_back(elem55);
    Elem46.push_back(elem46);
    Elem56.push_back(elem56);
    Elem66.push_back(elem66);
    Elem57.push_back(elem57);
    Elem67.push_back(elem67);
    Elem77.push_back(elem77);
    nrlines++;
  }
  dbdata.close();
  std::ofstream myMatrixFile("goodMatrix2008_09_02.dat",std::ios::out);
 
  for(int i=0; i<nrlines-1;++i){
    if (Elem33[i]>0.0 && Elem33[i]<30.0 && Elem34[i]<10.0 && Elem44[i]>0.0 && Elem44[i]<30.0 && Elem35[i]<20. && Elem45[i]<10. && Elem55[i]>0.0 && Elem55[i]<20.0 && Elem46[i]>-20. && Elem56[i]<20. && Elem66[i]>0.0 && Elem66[i]<20.0 && Elem57[i]>-20. && Elem67[i]<10.0 && Elem77[i]>0.0 && Elem77[i]<20.0){
      myMatrixFile<<index_id[i]<<"  "<<Elem33[i]<<"  "<<Elem34[i]<<"  "<<Elem44[i]<<"  "<<Elem35[i]<<"  "<<Elem45[i]<<"  "<<Elem55[i]<<"  "<<Elem46[i]<<"  "<<Elem56[i]<<"  "<<Elem66[i]<<"  "<<Elem57[i]<<"  "<<Elem67[i]<<"  "<<Elem77[i]<<std::endl;
    }
  }
}

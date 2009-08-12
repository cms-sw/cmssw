#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

int main(){

  const int MAX_SIZE = 252288;

  int old_index;
  float old_ped, old_rms;
  std::vector<int> old_index_id;
  std::vector<float> old_peds;
  std::vector<float> old_pedrms;
  int new_index;
  float new_ped,new_rms;
  std::vector<int> new_index_id;
  std::vector<float> new_peds;
  std::vector<float> new_pedrms;
  std::vector<float> diff;
  std::vector<float> myoldpeds;

  int counter,counter1;
  int old_nrlines=0;
  int new_nrlines=0;

  std::ifstream olddata; 
  olddata.open("goodPeds2008_09_02.dat",std::ios::in); 
  if(!olddata) {
    std::cerr <<"Error: goodPeds2008_09_02.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!olddata.eof() ) { 
    olddata >> old_index >> old_ped >> old_rms ; 
    old_index_id.push_back(old_index);
    old_peds.push_back(old_ped);
    old_pedrms.push_back(old_rms);
    old_nrlines++;
  }
  olddata.close();

  std::ifstream newdata;
  std::ofstream myPedsFile("diffPedsOct_Feb.dat",std::ios::out);
  newdata.open("goodPeds2009_02_16.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: goodPeds2009_02_16.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_index >> new_ped >> new_rms ; 
    new_index_id.push_back(new_index);
    new_peds.push_back(new_ped);
    new_pedrms.push_back(new_rms);
    new_nrlines++;
  }
  newdata.close();
  diff.resize(MAX_SIZE);
  myoldpeds.resize(MAX_SIZE);
  
  for(int i=0; i<MAX_SIZE;++i){
    counter=old_index_id[i];  
    myoldpeds[i]=old_peds[i];

    for (int k=0;k<new_index_id.size()-1;k++){
      counter1=new_index_id[k];
      if(counter == counter1){
	diff[k]=old_peds[i] - new_peds[k];
	//std::cout<<old_peds[i]<<" new_peds[k]"<<new_peds[k]<<std::endl;
	myPedsFile<<counter<<"  "<<diff[k]<<std::endl;	
      }
    }
  }
}

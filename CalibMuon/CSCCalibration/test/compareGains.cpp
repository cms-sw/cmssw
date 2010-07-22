#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

int main(){

  const int MAX_SIZE = 252288;

  int old_index;
  float old_slope, old_int, old_chi2;
  std::vector<int> old_index_id;
  std::vector<float> old_gains;
  std::vector<float> old_intercept;
  std::vector<float> old_chi;

  int new_index;
  float new_slope,new_int, new_chi2;
  std::vector<int> new_index_id;
  std::vector<float> new_gains;
  std::vector<float> new_intercept;
  std::vector<float> new_chi;
 
  std::vector<float> diff;
  std::vector<float> myoldgains;

  int counter,counter1;
  int old_nrlines=0;
  int new_nrlines=0;

  std::ifstream olddata; 
  olddata.open("goodGains2008_09_02.dat",std::ios::in); 
  if(!olddata) {
    std::cerr <<"Error: goodGains2008_09_02.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!olddata.eof() ) { 
    olddata >> old_index >> old_slope >> old_int >> old_chi2 ; 
    old_index_id.push_back(old_index);
    old_gains.push_back(old_slope);
    old_intercept.push_back(old_int);
    old_chi.push_back(old_chi2);
    old_nrlines++;
  }
  olddata.close();

  std::ifstream newdata;
  std::ofstream myGainsFile("diffGainsOct_Aug109889.dat",std::ios::out);
  newdata.open("goodGains2009_08_07_run109889.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: goodGains2009_08_07_run109889.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_index >> new_slope >> new_int >> new_chi2 ; 
    new_index_id.push_back(new_index);
    new_gains.push_back(new_slope);
    new_intercept.push_back(new_int);
    new_chi.push_back(new_chi2);
    new_nrlines++;
  }
  newdata.close();
  diff.resize(MAX_SIZE);
  myoldgains.resize(MAX_SIZE);
  
  for(int i=0; i<MAX_SIZE;++i){
    counter=old_index_id[i];  
    myoldgains[i]=old_gains[i];

    for (int k=0;k<new_index_id.size()-1;k++){
      counter1=new_index_id[k];
      if(counter == counter1){
	diff[k]=old_gains[i] - new_gains[k];
	//std::cout<<old_gains[i]<<" new_gains[k]"<<new_gains[k]<<std::endl;
	myGainsFile<<counter<<"  "<<diff[k]<<std::endl;	
      }
    }
  }
}

#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

int main(){

  const int MAX_SIZE = 252288;

  int old_index;
  float old_xtalk_left, old_xtalk_right, old_int_left, old_int_right;
  std::vector<int> old_index_id;
  std::vector<float> old_Rxtalk;
  std::vector<float> old_Lxtalk;
  std::vector<float> old_Rint;
  std::vector<float> old_Lint;
  std::vector<float> diffXtalkR;
  std::vector<float> diffXtalkL;
  std::vector<float> diffIntR;
  std::vector<float> diffIntL;
  
  std::vector<float> myoldxtalkR;
  std::vector<float> myoldxtalkL;
  std::vector<float> myoldintR;
  std::vector<float> myoldintL;

  int new_index,extra1,extra2;
  float new_xtalk_left, new_xtalk_right, new_int_left, new_int_right;
  std::vector<int> new_index_id;
  std::vector<float> new_Rxtalk;
  std::vector<float> new_Lxtalk;
  std::vector<float> new_Rint;
  std::vector<float> new_Lint;

  int counter,counter1;
  int old_nrlines=0;
  int new_nrlines=0;

   std::ifstream olddata; 
  olddata.open("goodXtalk2008_09_02.dat",std::ios::in); 
  if(!olddata) {
    std::cerr <<"Error: goodXtalk2008_09_02.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while (!olddata.eof() ) { 
    olddata >> old_index >> old_xtalk_left >> old_int_left >> old_xtalk_right >> old_int_right ;
    old_index_id.push_back(old_index);
    old_Rxtalk.push_back(old_xtalk_right);
    old_Rint.push_back(old_int_right); 
    old_Lxtalk.push_back(old_xtalk_left);  
    old_Lint.push_back(old_int_left);
  }
  olddata.close();

  std::ifstream newdata;
  std::ofstream myXtalkFile("diffXtalkOct_Aug3.dat",std::ios::out);

  newdata.open("goodXtalk2009_08_07_run109890.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error:goodXtalk2009_08_07_run109890.dat  -> no such file!"<< std::endl;
    exit(1);
  }

  while (!newdata.eof() ) { 
    newdata >> new_index >> new_xtalk_left >> new_int_left >> new_xtalk_right >> new_int_right ;
    new_index_id.push_back(new_index);
    new_Rxtalk.push_back(new_xtalk_right);
    new_Rint.push_back(new_int_right); 
    new_Lxtalk.push_back(new_xtalk_left);
    new_Lint.push_back(new_int_left);
  }
  newdata.close();  

  diffXtalkR.resize(MAX_SIZE);
  diffXtalkL.resize(MAX_SIZE);
  diffIntR.resize(MAX_SIZE);
  diffIntL.resize(MAX_SIZE);
  myoldxtalkR.resize(MAX_SIZE);
  myoldxtalkL.resize(MAX_SIZE);
  myoldintR.resize(MAX_SIZE);
  myoldintR.resize(MAX_SIZE);

  for(int i=0; i<MAX_SIZE;++i){
    counter=old_index_id[i];  
    myoldxtalkR[i]=old_Rxtalk[i];
    myoldxtalkL[i]=old_Lxtalk[i];
    myoldintR[i]=old_Rint[i];
    myoldintR[i]=old_Lint[i];

    for (int k=0;k<new_index_id.size()-1;k++){
      counter1=new_index_id[k];
      if(counter == counter1){
	diffXtalkR[k]=old_Rxtalk[i] - new_Rxtalk[k];
	diffXtalkL[k]=old_Lxtalk[i] - new_Lxtalk[k];
	diffIntR[k]=old_Rint[i] - new_Rint[k];
	diffIntL[k]=old_Rint[i] - new_Rint[k];

	//	std::cout<<counter<<" "<<counter1<<"  "<<old_Rxtalk[i]<<" new_Rxtalk[k] "<<new_Rxtalk[k]<<std::endl;
	myXtalkFile<<counter<<"  "<<diffXtalkL[k]<<"  "<<diffIntL[k]<<"  "<<diffXtalkR[k]<<"  "<<diffIntR[k]<<"  "<<std::endl;	
      }
    }
  }
}

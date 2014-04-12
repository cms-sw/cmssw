#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

int main(){

  const int MAX_SIZE = 252288;

  int old_index;
  float old_elem33, old_elem34,old_elem35,old_elem44,old_elem45,old_elem46,old_elem55,old_elem56;
  float old_elem57,old_elem66,old_elem67,old_elem77;
  std::vector<int> old_index_id;
  std::vector<float> old_el33;
  std::vector<float> old_el34;
  std::vector<float> old_el35;
  std::vector<float> old_el44;
  std::vector<float> old_el45;
  std::vector<float> old_el46;
  std::vector<float> old_el55;
  std::vector<float> old_el56;
  std::vector<float> old_el57;
  std::vector<float> old_el66;
  std::vector<float> old_el67;
  std::vector<float> old_el77;

  int new_index;
  float new_elem33,new_elem34,new_elem35,new_elem44,new_elem45,new_elem46,new_elem55,new_elem56;
  float new_elem57,new_elem66,new_elem67,new_elem77;
  std::vector<int> new_index_id;
  std::vector<float> new_el33;
  std::vector<float> new_el34;
  std::vector<float> new_el35;
  std::vector<float> new_el44;
  std::vector<float> new_el45;
  std::vector<float> new_el46;
  std::vector<float> new_el55;
  std::vector<float> new_el56;
  std::vector<float> new_el57;
  std::vector<float> new_el66;
  std::vector<float> new_el67;
  std::vector<float> new_el77;

  //differences
  std::vector<float> diff_el33;
  std::vector<float> diff_el34;
  std::vector<float> diff_el35;
  std::vector<float> diff_el44;
  std::vector<float> diff_el45;
  std::vector<float> diff_el46;
  std::vector<float> diff_el55;
  std::vector<float> diff_el56;
  std::vector<float> diff_el57;
  std::vector<float> diff_el66;
  std::vector<float> diff_el67;
  std::vector<float> diff_el77;

  //old vectors
  std::vector<float> myoldel33;
  std::vector<float> myoldel34;
  std::vector<float> myoldel35;
  std::vector<float> myoldel44;
  std::vector<float> myoldel45;
  std::vector<float> myoldel46;
  std::vector<float> myoldel55;
  std::vector<float> myoldel56;
  std::vector<float> myoldel57;
  std::vector<float> myoldel66;
  std::vector<float> myoldel67;
  std::vector<float> myoldel77;

  int counter,counter1;
  int old_nrlines=0;
  int new_nrlines=0;

  std::ifstream olddata; 
  olddata.open("goodMatrix2008_09_02.dat",std::ios::in); 
  if(!olddata) {
    std::cerr <<"Error: goodMatrix2008_09_02.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!olddata.eof() ) { 
    olddata >> old_index >> old_elem33 >> old_elem34 >> old_elem44 >> old_elem35 >> old_elem45 >> old_elem55 >> old_elem46 >> old_elem56 >>old_elem66 >> old_elem57 >> old_elem67 >> old_elem77 ; 
    old_index_id.push_back(old_index);
    old_el33.push_back(old_elem33);
    old_el34.push_back(old_elem34);
    old_el35.push_back(old_elem35);
    old_el44.push_back(old_elem44);
    old_el45.push_back(old_elem45);
    old_el46.push_back(old_elem46);
    old_el55.push_back(old_elem55);
    old_el56.push_back(old_elem56);
    old_el57.push_back(old_elem57);
    old_el66.push_back(old_elem66);
    old_el67.push_back(old_elem67);
    old_el77.push_back(old_elem77);
    old_nrlines++;
  }
  olddata.close();

  std::ifstream newdata;
  std::ofstream myXtalkFile("diffMatrixOct_Aug109891.dat",std::ios::out);
  newdata.open("goodMatrix2009_08_07_run109891.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: goodMatrix2009_08_07_run109891.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_index >> new_elem33 >> new_elem34 >> new_elem44 >> new_elem35 >> new_elem45 >> new_elem55 >> new_elem46 >> new_elem56 >>new_elem66 >> new_elem57 >> new_elem67 >> new_elem77; 
    new_index_id.push_back(new_index);
    new_el33.push_back(new_elem33);
    new_el34.push_back(new_elem34);
    new_el35.push_back(new_elem35);
    new_el44.push_back(new_elem44);
    new_el45.push_back(new_elem45);
    new_el46.push_back(new_elem46);
    new_el55.push_back(new_elem55);
    new_el56.push_back(new_elem56);
    new_el57.push_back(new_elem57);
    new_el66.push_back(new_elem66);
    new_el67.push_back(new_elem67);
    new_el77.push_back(new_elem77);
    new_nrlines++;
  }
  newdata.close();

  //resize
  diff_el33.resize(MAX_SIZE);
  diff_el34.resize(MAX_SIZE);
  diff_el35.resize(MAX_SIZE);
  diff_el44.resize(MAX_SIZE);
  diff_el45.resize(MAX_SIZE);
  diff_el46.resize(MAX_SIZE);
  diff_el55.resize(MAX_SIZE);
  diff_el56.resize(MAX_SIZE);
  diff_el57.resize(MAX_SIZE);
  diff_el66.resize(MAX_SIZE);
  diff_el67.resize(MAX_SIZE);
  diff_el77.resize(MAX_SIZE);

  myoldel33.resize(MAX_SIZE);
  myoldel34.resize(MAX_SIZE);
  myoldel35.resize(MAX_SIZE);
  myoldel44.resize(MAX_SIZE);
  myoldel45.resize(MAX_SIZE);
  myoldel46.resize(MAX_SIZE);
  myoldel55.resize(MAX_SIZE);
  myoldel56.resize(MAX_SIZE);
  myoldel57.resize(MAX_SIZE);
  myoldel66.resize(MAX_SIZE);
  myoldel67.resize(MAX_SIZE);
  myoldel77.resize(MAX_SIZE);

  for(int i=0; i<MAX_SIZE;++i){
    counter=old_index_id[i];  
    myoldel33[i]=old_el33[i];
    myoldel34[i]=old_el34[i];
    myoldel35[i]=old_el35[i];
    myoldel44[i]=old_el44[i];
    myoldel45[i]=old_el45[i];
    myoldel46[i]=old_el46[i];
    myoldel55[i]=old_el55[i];
    myoldel56[i]=old_el56[i];
    myoldel57[i]=old_el57[i];
    myoldel66[i]=old_el66[i];
    myoldel67[i]=old_el67[i];
    myoldel77[i]=old_el77[i];
    
    for (int k=0;k<new_index_id.size()-1;k++){
      counter1=new_index_id[k];
      if(counter == counter1){
	diff_el33[k]=old_el33[i] - new_el33[k];
	diff_el34[k]=old_el34[i] - new_el34[k];
	diff_el35[k]=old_el35[i] - new_el35[k];
	diff_el44[k]=old_el44[i] - new_el44[k];
	diff_el45[k]=old_el45[i] - new_el45[k];
	diff_el46[k]=old_el46[i] - new_el46[k];
	diff_el55[k]=old_el55[i] - new_el55[k];
	diff_el56[k]=old_el56[i] - new_el56[k];
	diff_el57[k]=old_el57[i] - new_el57[k];
	diff_el66[k]=old_el66[i] - new_el66[k];
	diff_el67[k]=old_el67[i] - new_el67[k];
	diff_el77[k]=old_el77[i] - new_el77[k];
	//std::cout<<old_el33[i]<<" new_el33[k]"<<new_el33[k]<<std::endl;
	myXtalkFile<<counter<<"  "<<diff_el33[k]<<"  "<< diff_el34[k]<<"  "<<diff_el35[k]<<"  "<<diff_el44[k]<<"  "<<diff_el45[k]<<"  "<<diff_el46[k]<<"  "<<diff_el55[k]<<"  "<<diff_el56[k]<<"  "<<diff_el57[k]<<"  "<<diff_el66[k]<<"  "<<diff_el67[k]<<"  "<<diff_el77[k]<<"  "<<std::endl;	
      }
    }
  }
}

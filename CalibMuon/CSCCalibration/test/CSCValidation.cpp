#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <memory>

int main()
{
  //Gains variables
  int counter=0;
  int dbgains_nrlines=0;
  int fakegains_nrlines=0;
 
  int dbgains_index;
  float db_gainslope;
  std::vector<int> dbgains_index_id;
  std::vector<float> db_slope;
  int fakegains_index;
  float fake_gainslope;
  std::vector<int> fakegains_index_id;
  std::vector<float> fake_slope;

  //NoiseMatrix variables
  int dbmatrix_nrlines=0;
  int fakematrix_nrlines=0;
  int fakematrix_index;
  int dbmatrix_index;
  float db_elm33,db_elm34, db_elm44, db_elm35, db_elm45, db_elm55;
  float db_elm46, db_elm56, db_elm66, db_elm57, db_elm67, db_elm77;
  std::vector<int> dbmatrix_index_id;
  std::vector<float> db_elem33;
  std::vector<float> db_elem34;
  std::vector<float> db_elem44;
  std::vector<float> db_elem45;
  std::vector<float> db_elem35;
  std::vector<float> db_elem55;
  std::vector<float> db_elem46;
  std::vector<float> db_elem56;
  std::vector<float> db_elem66;
  std::vector<float> db_elem57;
  std::vector<float> db_elem67;
  std::vector<float> db_elem77;
  float fake_elm33,fake_elm34, fake_elm44, fake_elm35, fake_elm45, fake_elm55;
  float fake_elm46, fake_elm56, fake_elm66, fake_elm57, fake_elm67, fake_elm77;
  std::vector<int> fakematrix_index_id;
  std::vector<float> fake_elem33;
  std::vector<float> fake_elem34;
  std::vector<float> fake_elem44;
  std::vector<float> fake_elem45;
  std::vector<float> fake_elem35;
  std::vector<float> fake_elem55;
  std::vector<float> fake_elem46;
  std::vector<float> fake_elem56;
  std::vector<float> fake_elem66;
  std::vector<float> fake_elem57;
  std::vector<float> fake_elem67;
  std::vector<float> fake_elem77;

  //Pedestal variables
  int dbpeds_nrlines=0;
  int fakepeds_nrlines=0;
  int dbpeds_index,fakepeds_index;
  float fake_peds,fake_rms;
  std::vector<int> fakepeds_index_id;
  std::vector<float> fake_pedestal;
  std::vector<float> fake_pedrms;
  float db_peds,db_rms;
  std::vector<int> dbpeds_index_id;
  std::vector<float> db_pedestal;
  std::vector<float> db_pedrms;

  //Crosstalk variables
  int dbxtalk_nrlines=0;
  int fakextalk_nrlines=0;
  int dbxtalk_index,fakextalk_index;
  float dbxtalk_slope_right,dbxtalk_slope_left,dbxtalk_intercept_right;
  float dbxtalk_intercept_left; 
  float fakextalk_slope_right,fakextalk_slope_left,fakextalk_intercept_right;
  float fakextalk_intercept_left; 
  std::vector<int> fakextalk_index_id;
  std::vector<float> fakextalk_slope_r;
  std::vector<float> fakextalk_intercept_r;
  std::vector<float> fakextalk_slope_l;
  std::vector<float> fakextalk_intercept_l;
  std::vector<int> dbxtalk_index_id;
  std::vector<float> dbxtalk_slope_r;
  std::vector<float> dbxtalk_intercept_r;
  std::vector<float> dbxtalk_slope_l;
  std::vector<float> dbxtalk_intercept_l;

  /////////////////////////////////////////////////////////////////////////////////////////////
  //read fakes-on-the-fly for Gains
  std::ifstream fakegainsdata; 
  fakegainsdata.open("fakegains.dat",std::ios::in); 
  if(!fakegainsdata) {
    std::cerr <<"Error: fakegains.dat -> no such file!"<< std::endl;
    exit(1);
  }
  while (!fakegainsdata.eof() ) { 
    fakegainsdata >> fakegains_index >> fake_gainslope; 
    fakegains_index_id.push_back(fakegains_index);
    fake_slope.push_back(fake_gainslope);
    fakegains_nrlines++;
  }
  fakegainsdata.close();

  //read database values for Gains
  std::ifstream dbgainsdata; 
  dbgainsdata.open("dbgains.dat",std::ios::in); 
  if(!dbgainsdata) {
    std::cerr <<"Error: dbgains.dat -> no such file!"<< std::endl;
    exit(1);
  }
  while (!dbgainsdata.eof() ) { 
    dbgainsdata >> dbgains_index >> db_gainslope; 
    dbgains_index_id.push_back(dbgains_index);
    db_slope.push_back(db_gainslope);
    dbgains_nrlines++;
  }
  dbgainsdata.close();

 
  for(int i=0; i<217728; i++){
    if(fake_slope[i] != db_slope[i]){
      std::cout<<"ERROR::: CSC Gains object::: Values in DB incompatible with Fakes!"<<std::endl;    
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  //read fakes-on-the-fly for NoiseMatrix
  std::ifstream fakematrixdata; 
  fakematrixdata.open("fakematrix.dat",std::ios::in); 
  if(!fakematrixdata) {
    std::cerr <<"Error: fakematrix.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!fakematrixdata.eof() ) { 
    fakematrixdata >> fakematrix_index >> fake_elm33 >> fake_elm34 >> fake_elm44 >> fake_elm35 >> fake_elm45 >> fake_elm55 >> fake_elm46 >> fake_elm56 >> fake_elm66 >> fake_elm57 >> fake_elm67 >> fake_elm77; 
    fakematrix_index_id.push_back(fakematrix_index);
    fake_elem33.push_back(fake_elm33);
    fake_elem34.push_back(fake_elm34);
    fake_elem44.push_back(fake_elm44);
    fake_elem35.push_back(fake_elm35);
    fake_elem45.push_back(fake_elm45);
    fake_elem55.push_back(fake_elm55);
    fake_elem46.push_back(fake_elm46);
    fake_elem56.push_back(fake_elm56);
    fake_elem66.push_back(fake_elm66);
    fake_elem57.push_back(fake_elm57);
    fake_elem67.push_back(fake_elm67);
    fake_elem77.push_back(fake_elm77);

    fakematrix_nrlines++;
  }
  fakematrixdata.close();

  //read database values for NoiseMatrix
  std::ifstream dbmatrixdata;
  dbmatrixdata.open("dbmatrix.dat",std::ios::in); 
  if(!dbmatrixdata) {
    std::cerr <<"Error: dbmatrix.txt -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbmatrixdata.eof() ) { 
    dbmatrixdata >> dbmatrix_index >> db_elm33 >> db_elm34 >> db_elm44 >> db_elm35 >> db_elm45 >> db_elm55 >> db_elm46 >> db_elm56 >> db_elm66 >> db_elm57 >> db_elm67 >> db_elm77 ; 
    dbmatrix_index_id.push_back(dbmatrix_index);
    db_elem33.push_back(db_elm33);
    db_elem34.push_back(db_elm34);
    db_elem44.push_back(db_elm44);
    db_elem35.push_back(db_elm35);
    db_elem45.push_back(db_elm45);
    db_elem55.push_back(db_elm55);
    db_elem46.push_back(db_elm46);
    db_elem56.push_back(db_elm56);
    db_elem66.push_back(db_elm66);
    db_elem57.push_back(db_elm57);
    db_elem67.push_back(db_elm67);
    db_elem77.push_back(db_elm77); 
    dbmatrix_nrlines++;
  }
  dbmatrixdata.close();


  for(int i=0; i<217728; i++){
    if(fake_elem33[i]  != db_elem33[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem33 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem34[i]  != db_elem34[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem34 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem44[i]  != db_elem44[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem44 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem35[i]  != db_elem35[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem35 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem45[i]  != db_elem45[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem45 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem55[i]  != db_elem55[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem55 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem46[i]  != db_elem46[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem46 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem56[i]  != db_elem56[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem56 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem66[i]  != db_elem66[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem66 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem57[i]  != db_elem57[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem57 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem67[i]  != db_elem67[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem67 Values in DB incompatible with Fakes!"<<std::endl;
    }
    if(fake_elem77[i]  != db_elem77[i]){
      std::cout<<"ERROR::: CSC NoiseMatrix object:::elem77 Values in DB incompatible with Fakes!"<<std::endl;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  //read fakes_on-the-fly for Pedestals
  std::ifstream fakepedsdata; 
  fakepedsdata.open("fakepeds.dat",std::ios::in); 
  if(!fakepedsdata) {
    std::cerr <<"Error: fakepeds.dat -> no such file!"<< std::endl;
    exit(1);
  }
  while (!fakepedsdata.eof() ) { 
    fakepedsdata >> fakepeds_index >> fake_peds >> fake_rms; 
    fakepeds_index_id.push_back(fakepeds_index);
    fake_pedestal.push_back(fake_peds);
    fake_pedrms.push_back(fake_rms);
    fakepeds_nrlines++;
  }
  fakepedsdata.close();

  //read database values for Pedestals
  std::ifstream dbpedsdata; 
  dbpedsdata.open("dbpeds.dat",std::ios::in); 
  if(!dbpedsdata) {
    std::cerr <<"Error: dbpeds.dat -> no such file!"<< std::endl;
    exit(1);
  }
  while (!dbpedsdata.eof() ) { 
    dbpedsdata >> dbpeds_index >> db_peds >> db_rms ; 
    dbpeds_index_id.push_back(dbpeds_index);
    db_pedestal.push_back(db_peds);
    db_pedrms.push_back(db_rms);
    dbpeds_nrlines++;
  }
  dbpedsdata.close();

 
  for(int i=0; i<217728; i++){
    if(fake_slope[i] != db_slope[i]){
      std::cout<<"ERROR::: CSC Pedestal object::: Values in DB incompatible with Fakes!"<<std::endl;    
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //read fakes-on-the-fly for Crosstalk
  std::ifstream fakextalkdata; 
  fakextalkdata.open("fakextalk.dat",std::ios::in); 
  if(!fakextalkdata) {
    std::cerr <<"Error: fakextalk.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!fakextalkdata.eof() ) { 
    fakextalkdata >> fakextalk_index >> fakextalk_slope_right >> fakextalk_intercept_right >> fakextalk_slope_left >> fakextalk_intercept_left;
    fakextalk_index_id.push_back(fakextalk_index);
    fakextalk_slope_r.push_back(fakextalk_slope_right);
    fakextalk_slope_l.push_back(fakextalk_slope_left);
    fakextalk_intercept_r.push_back(fakextalk_intercept_right);
    fakextalk_intercept_l.push_back(fakextalk_intercept_left);
    fakextalk_nrlines++;
  }
  fakextalkdata.close();

  //read database values for Crosstalk
  std::ifstream dbxtalkdata;
  dbxtalkdata.open("dbxtalk.dat",std::ios::in); 
  if(!dbxtalkdata) {
    std::cerr <<"Error: dbxtalk.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbxtalkdata.eof() ) { 
    dbxtalkdata >> dbxtalk_index >> dbxtalk_slope_right >> dbxtalk_intercept_right >> dbxtalk_slope_left >> dbxtalk_intercept_left ; 
    dbxtalk_index_id.push_back(dbxtalk_index);
    dbxtalk_slope_r.push_back(dbxtalk_slope_right);
    dbxtalk_slope_l.push_back(dbxtalk_slope_left);
    dbxtalk_intercept_r.push_back(dbxtalk_intercept_right);
    dbxtalk_intercept_l.push_back(dbxtalk_intercept_left);
    dbxtalk_nrlines++;
  }
  dbxtalkdata.close();

  for(int i=0; i<217728; i++){
    if(fakextalk_slope_r[i] != dbxtalk_slope_r[i]){
      //      std::cout<<fakextalk_slope_r[i]<<"  "<<dbxtalk_slope_r[i]<<std::endl;
      std::cout<<"ERROR::: CSC Crosstalk object:::Slope_right Values in DB incompatible with Fakes!"<<std::endl;    
    }
    if(fakextalk_slope_l[i] != dbxtalk_slope_l[i]){
      std::cout<<"ERROR::: CSC Crosstalk object:::Slope_left Values in DB incompatible with Fakes!"<<std::endl;    
    }
    if(fakextalk_intercept_r[i] != dbxtalk_intercept_r[i]){
      std::cout<<"ERROR::: CSC Crosstalk object:::Intercept_right Values in DB incompatible with Fakes!"<<std::endl;    
    }
    if(fakextalk_intercept_l[i] != dbxtalk_intercept_l[i]){
      std::cout<<"ERROR::: CSC Crosstalk object:::Intercept_left Values in DB incompatible with Fakes!"<<std::endl;    
    }
  }
}

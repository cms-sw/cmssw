#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkDBConditions.h"

void CSCCrosstalkDBConditions::prefillDBCrosstalk()
{
  cndbcrosstalk = new CSCDBCrosstalk();
  //const CSCDetId& detId = CSCDetId();
      
  int counter;
  int db_nrlines=0;
  int new_nrlines=0;
  seed = 10000;	
  srand(seed);
  mean=6.8, min=-10.0, minchi=1.0, M=1000;
  
  std::ifstream dbdata; 
  dbdata.open("dbxtalk.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: dbxtalk.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbdata.eof() ) { 
    dbdata >> db_index >> db_slope_right >> db_intercept_right >> db_chi2_right >> db_slope_left >> db_intercept_left >> db_chi2_left ;
    db_index_id.push_back(db_index);
    db_slope_r.push_back(db_slope_right);
    db_slope_l.push_back(db_slope_left);
    db_intercept_r.push_back(db_intercept_right);
    db_intercept_l.push_back(db_intercept_left);
    db_chi2_r.push_back(db_chi2_right);
    db_chi2_l.push_back(db_chi2_left); 
    db_nrlines++;
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("xtalk.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: xtalk.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_chamber_id >> new_index>> new_strip >> new_slope_right >> new_intercept_right >> new_chi2_right >> new_slope_left >> new_intercept_left >> new_chi2_left ; 
    new_cham_id.push_back(new_chamber_id);
    new_index_id.push_back(new_index);
    new_strips.push_back(new_strip);
    new_slope_r.push_back(new_slope_right);
    new_slope_l.push_back(new_slope_left);
    new_intercept_r.push_back(new_intercept_right);
    new_intercept_l.push_back(new_intercept_left);
    new_chi2_r.push_back(new_chi2_right);
    new_chi2_l.push_back(new_chi2_left);
    new_nrlines++;
  }
  newdata.close();
  
  std::vector<CSCDBCrosstalk::Item> itemvector;
   itemvector.resize(217728);

  for(int i=0; i<217728;++i){
    itemvector[i].xtalk_slope_right=db_slope_r[i];
    itemvector[i].xtalk_intercept_right=db_intercept_r[i]; 
    itemvector[i].xtalk_chi2_right=db_chi2_r[i];
    itemvector[i].xtalk_slope_left=db_slope_l[i];  
    itemvector[i].xtalk_intercept_left=db_intercept_l[i];  
    itemvector[i].xtalk_chi2_left=db_chi2_l[i];
  }

  for(int i=0; i<217728;++i){
    counter=db_index_id[i];  
    for (unsigned int k=0;k<new_index_id.size()-1;k++){
      if(counter==new_index_id[k]){
	itemvector[counter].xtalk_slope_right=db_slope_r[k];
	itemvector[counter].xtalk_intercept_right=db_intercept_r[k]; 
	itemvector[counter].xtalk_chi2_right=db_chi2_r[k];
	itemvector[counter].xtalk_slope_left=db_slope_l[k];  
	itemvector[counter].xtalk_intercept_left=db_intercept_l[k];  
	itemvector[counter].xtalk_chi2_left=db_chi2_l[k];
	itemvector[i] = itemvector[counter];
      }  
    }
  }
  
  std::copy(itemvector.begin(), itemvector.end(), std::back_inserter(cndbcrosstalk->crosstalk));
}
  

CSCCrosstalkDBConditions::CSCCrosstalkDBConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBCrosstalk();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCCrosstalkDBConditions::produceDBCrosstalk);
  findingRecord<CSCDBCrosstalkRcd>();
  //now do what ever other initialization is needed
}


CSCCrosstalkDBConditions::~CSCCrosstalkDBConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbcrosstalk;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCCrosstalkDBConditions::ReturnType
CSCCrosstalkDBConditions::produceDBCrosstalk(const CSCDBCrosstalkRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCDBCrosstalk* mydata=new CSCDBCrosstalk( *cndbcrosstalk );
  return mydata;
  
}

 void CSCCrosstalkDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }

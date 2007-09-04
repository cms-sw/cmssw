#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCGainsDBConditions.h"

void CSCGainsDBConditions::prefillDBGains()
{
  cndbgains = new CSCDBGains();
  //const CSCDetId& detId = CSCDetId();
      
  int counter;
  int db_nrlines=0;
  int new_nrlines=0;
  seed = 10000;	
  srand(seed);
  mean=6.8, min=-10.0, minchi=1.0, M=1000;
  
  std::ifstream dbdata; 
  dbdata.open("dbgains.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: dbgains.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbdata.eof() ) { 
    dbdata >> db_index >> db_gainslope >> db_intercpt >> db_chisq ; 
    db_index_id.push_back(db_index);
    db_slope.push_back(db_gainslope);
    db_intercept.push_back(db_intercpt);
    db_chi2.push_back(db_chisq);
    db_nrlines++;
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("gains.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: gains.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_chamber_id >> new_index>> new_strip >> new_gainslope >> new_intercpt >> new_chisq ; 
    new_cham_id.push_back(new_chamber_id);
    new_index_id.push_back(new_index);
    new_strips.push_back(new_strip);
    new_slope.push_back(new_gainslope);
    new_intercept.push_back(new_intercpt);
    new_chi2.push_back(new_chisq);
    new_nrlines++;
  }
  newdata.close();
  
  std::vector<CSCDBGains::Item> itemvector;
   itemvector.resize(217728);

  for(int i=0; i<217728;++i){
    itemvector[i].gain_slope= db_slope[i];
    itemvector[i].gain_intercept= db_intercept[i];
    itemvector[i].gain_chi2= db_chi2[i];
  }

  for(int i=0; i<217728;++i){
    counter=db_index_id[i];  
    for (unsigned int k=0;k<new_index_id.size()-1;k++){
      if(counter==new_index_id[k]){
	itemvector[counter].gain_slope= new_slope[k];
	itemvector[counter].gain_intercept= new_intercept[k];
	itemvector[counter].gain_chi2= new_chi2[k];
	itemvector[i] = itemvector[counter];
      }  
    }
  }
  
  std::copy(itemvector.begin(), itemvector.end(), std::back_inserter(cndbgains->gains));
}
  

CSCGainsDBConditions::CSCGainsDBConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBGains();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCGainsDBConditions::produceDBGains);
  findingRecord<CSCDBGainsRcd>();
  //now do what ever other initialization is needed
}


CSCGainsDBConditions::~CSCGainsDBConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbgains;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCGainsDBConditions::ReturnType
CSCGainsDBConditions::produceDBGains(const CSCDBGainsRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCDBGains* mydata=new CSCDBGains( *cndbgains );
  return mydata;
  
}

 void CSCGainsDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }

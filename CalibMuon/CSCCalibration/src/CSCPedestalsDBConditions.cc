#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCPedestalsDBConditions.h"

void CSCPedestalsDBConditions::prefillDBPedestals()
{
  cndbpedestals = new CSCDBPedestals();
  //const CSCDetId& detId = CSCDetId();
      
  int counter;
  int db_nrlines=0;
  int new_nrlines=0;
  seed = 10000;	
  srand(seed);
  mean=6.8, min=-10.0, minchi=1.0, M=1000;
  
  std::ifstream dbdata; 
  dbdata.open("dbpeds.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: dbpeds.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbdata.eof() ) { 
    dbdata >> db_index >> db_ped >> db_rms ; 
    db_index_id.push_back(db_index);
    db_peds.push_back(db_ped);
    db_pedrms.push_back(db_rms);
    db_nrlines++;
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("new_peds.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: new_peds.dat -> no such file!"<< std::endl;
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
  
  std::vector<CSCDBPedestals::Item> itemvector;
  itemvector.resize(252288);
  // itemvector.resize(217728);

  //for(int i=0; i<217728;++i){
  for(int i=0; i<252288;++i){
    itemvector[i].ped= db_peds[i];
    itemvector[i].rms= db_pedrms[i];
  }

  //for(int i=0; i<217728;++i){
  for(int i=0; i<252288;++i){
     counter=db_index_id[i];  
     for (unsigned int k=0;k<new_index_id.size()-1;k++){
       if(counter==new_index_id[k]){
	 itemvector[counter].ped= new_peds[k];
	 itemvector[counter].rms= new_pedrms[k];
	 itemvector[i] = itemvector[counter];
	//std::cout<<"counter "<<counter<<" new_index_id[k] "<<new_index_id[k]<<" new_slope[k] "<<new_slope[k]<<" db_slope[k] "<<db_slope[k]<<std::endl;
       }  
     }
   }
   
   std::copy(itemvector.begin(), itemvector.end(), std::back_inserter(cndbpedestals->pedestals));
}


CSCPedestalsDBConditions::CSCPedestalsDBConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBPedestals();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCPedestalsDBConditions::produceDBPedestals);
  findingRecord<CSCDBPedestalsRcd>();
  //now do what ever other initialization is needed
}


CSCPedestalsDBConditions::~CSCPedestalsDBConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbpedestals;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCPedestalsDBConditions::ReturnType
CSCPedestalsDBConditions::produceDBPedestals(const CSCDBPedestalsRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCDBPedestals* mydata=new CSCDBPedestals( *cndbpedestals );
  return mydata;
  
}

 void CSCPedestalsDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }

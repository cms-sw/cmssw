#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixDBConditions.h"

void CSCNoiseMatrixDBConditions::prefillDBNoiseMatrix()
{
  cndbmatrix = new CSCDBNoiseMatrix();
  //const CSCDetId& detId = CSCDetId();
      
  int counter;
  int db_nrlines=0;
  int new_nrlines=0;
    
  std::ifstream dbdata; 
  dbdata.open("dbmatrix.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: dbmatrix.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbdata.eof() ) { 
    dbdata >> db_index >> db_elm33 >> db_elm34 >> db_elm44 >> db_elm35 >> db_elm45 >> db_elm55 >> db_elm46 >> db_elm56 >> db_elm66 >> db_elm57 >> db_elm67 >> db_elm77 ; 
    db_index_id.push_back(db_index);
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
    db_nrlines++;
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("matrix.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: matrix.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_chamber_id >> new_index>> new_elm33 >> new_elm34 >> new_elm44 >> new_elm35 >> new_elm45 >> new_elm55 >> new_elm46 >> new_elm56 >> new_elm66 >> new_elm57 >> new_elm67 >> new_elm77 ; 
    new_cham_id.push_back(new_chamber_id);
    new_index_id.push_back(new_index);
    new_elem33.push_back(new_elm33);
    new_elem34.push_back(new_elm34);
    new_elem44.push_back(new_elm44);
    new_elem35.push_back(new_elm35);
    new_elem45.push_back(new_elm45);
    new_elem55.push_back(new_elm55);
    new_elem46.push_back(new_elm46);
    new_elem56.push_back(new_elm56);
    new_elem66.push_back(new_elm66);
    new_elem57.push_back(new_elm57);
    new_elem67.push_back(new_elm67);
    new_elem77.push_back(new_elm77); 
    new_nrlines++;
  }
  newdata.close();
  
  std::vector<CSCDBNoiseMatrix::Item> itemvector;
   itemvector.resize(217728);

  for(int i=0; i<217728;++i){
    itemvector[i].elem33 = db_elem33[i];
    itemvector[i].elem34 = db_elem34[i]; 
    itemvector[i].elem44 = db_elem44[i];
    itemvector[i].elem35 = db_elem35[i];
    itemvector[i].elem45 = db_elem45[i];
    itemvector[i].elem55 = db_elem55[i];
    itemvector[i].elem46 = db_elem46[i];
    itemvector[i].elem56 = db_elem56[i];
    itemvector[i].elem66 = db_elem66[i];
    itemvector[i].elem57 = db_elem57[i];
    itemvector[i].elem67 = db_elem67[i];
    itemvector[i].elem77 = db_elem77[i];
  }

  for(int i=0; i<217728;++i){
    counter=db_index_id[i];  
    for (unsigned int k=0;k<new_index_id.size()-1;k++){
      if(counter==new_index_id[k]){
	itemvector[counter].elem33 = db_elem33[k];
	itemvector[counter].elem34 = db_elem34[k]; 
	itemvector[counter].elem44 = db_elem44[k];
	itemvector[counter].elem35 = db_elem35[k];
	itemvector[counter].elem45 = db_elem45[k];
	itemvector[counter].elem55 = db_elem55[k];
	itemvector[counter].elem46 = db_elem46[k];
	itemvector[counter].elem56 = db_elem56[k];
	itemvector[counter].elem66 = db_elem66[k];
	itemvector[counter].elem57 = db_elem57[k];
	itemvector[counter].elem67 = db_elem67[k];
	itemvector[counter].elem77 = db_elem77[k];
	itemvector[i] = itemvector[counter];
      }  
    }
  }
  
  std::copy(itemvector.begin(), itemvector.end(), std::back_inserter(cndbmatrix->matrix));
}
  

CSCNoiseMatrixDBConditions::CSCNoiseMatrixDBConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBNoiseMatrix();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCNoiseMatrixDBConditions::produceDBNoiseMatrix);
  findingRecord<CSCDBNoiseMatrixRcd>();
  //now do what ever other initialization is needed
}


CSCNoiseMatrixDBConditions::~CSCNoiseMatrixDBConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbmatrix;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCNoiseMatrixDBConditions::ReturnType
CSCNoiseMatrixDBConditions::produceDBNoiseMatrix(const CSCDBNoiseMatrixRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCDBNoiseMatrix* mydata=new CSCDBNoiseMatrix( *cndbmatrix );
  return mydata;
  
}

 void CSCNoiseMatrixDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }

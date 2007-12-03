#include <memory>
#include <fstream>
#include "boost/shared_ptr.hpp"

#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalk.h"

void CSCFakeDBCrosstalk::prefillDBFakeCrosstalk(){
  
  cndbcrosstalk = new CSCDBCrosstalk();
  
  seed = 10000;	
  srand(seed);
  mean=-0.0009, min=0.035, minchi=1.5, M=1000;
  ii=0,jj=0,iii=0,jjj=0;  

  std::vector<CSCDBCrosstalk::Item> *itemvector;
  itemvector = new std::vector<CSCDBCrosstalk::Item> ;
  itemvector->resize(217728);

  
  for(int i=0; i<217728;++i){
    itemvector->at(i).xtalk_slope_right = -((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean;
    itemvector->at(i).xtalk_intercept_right= ((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min;
    itemvector->at(i).xtalk_chi2_right= ((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
    itemvector->at(i).xtalk_slope_left=-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean;
    itemvector->at(i).xtalk_intercept_left=((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min;
    itemvector->at(i).xtalk_chi2_left=((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
    
    //80 strips per chamber
    if(i<34561 && i%80==0){
    //if(i<217728 && i%80==0){
      itemvector->at(i).xtalk_slope_left=0.0;
      itemvector->at(i).xtalk_intercept_left=0.0;
      itemvector->at(i).xtalk_chi2_left=0.0;
    }
   
    if(i!=0 && i<34561 && (i+1)%80==0){
      itemvector->at(i).xtalk_slope_right=0.0;
      itemvector->at(i).xtalk_intercept_right=0.0;
      itemvector->at(i).xtalk_chi2_right=0.0;
    }
    
    //64 strips per chamber
    if(i>34560 && i<48385 && i%64==0){
      itemvector->at(i).xtalk_slope_left=0.0;
      itemvector->at(i).xtalk_intercept_left=0.0;
      itemvector->at(i).xtalk_chi2_left=0.0;
    }

    if(i>34560 && i<48385 && (i+1)%64==0){
      itemvector->at(i).xtalk_slope_right=0.0;
      itemvector->at(i).xtalk_intercept_right=0.0;
      itemvector->at(i).xtalk_chi2_right=0.0;
    }
    
    //80 strips per chamber again
    if(i>48384 && i<143425){
      ii++;
      if(i>48384 && i<143425 && ii%80==0){
	itemvector->at(i).xtalk_slope_left=0.0;
	itemvector->at(i).xtalk_intercept_left=0.0;
	itemvector->at(i).xtalk_chi2_left=0.0;
      }
    }
    
    if(i>48384 && i<143425){
      jj++;
      if(i>48384 && i<143425 && (jj+1)%80==0){
	itemvector->at(i).xtalk_slope_right=0.0;
	itemvector->at(i).xtalk_intercept_right=0.0;
	itemvector->at(i).xtalk_chi2_right=0.0;
      }
    }
    
    //64 strips per chamber again
    if(i>143424 && i<157249 &&i%64==0){
      itemvector->at(i).xtalk_slope_left=0.0;
      itemvector->at(i).xtalk_intercept_left=0.0;
      itemvector->at(i).xtalk_chi2_left=0.0;
    }

    if(i>143424 && i<157249 && (i+1)%64==0){
      itemvector->at(i).xtalk_slope_right=0.0;
      itemvector->at(i).xtalk_intercept_right=0.0;
      itemvector->at(i).xtalk_chi2_right=0.0;
    }

    
    //80 strips per chamber last time
    if(i>157248){
      iii++;
      if(i>157248 && iii%80==0){
	itemvector->at(i).xtalk_slope_left=0.0;
	itemvector->at(i).xtalk_intercept_left=0.0;
	itemvector->at(i).xtalk_chi2_left=0.0;
      }
    }

    if(i>157248){
      jjj++;
      if(i>157248 && (jjj+1)%80==0){
	itemvector->at(i).xtalk_slope_right=0.0;
	itemvector->at(i).xtalk_intercept_right=0.0;
	itemvector->at(i).xtalk_chi2_right=0.0;
      }
    }
  }
  std::copy(itemvector->begin(), itemvector->end(), std::back_inserter(cndbcrosstalk->crosstalk)); 
  delete itemvector;
}


CSCFakeDBCrosstalk::CSCFakeDBCrosstalk(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBFakeCrosstalk();
  setWhatProduced(this,&CSCFakeDBCrosstalk::produceDBCrosstalk);
  findingRecord<CSCDBCrosstalkRcd>();
  //now do what ever other initialization is needed
}


CSCFakeDBCrosstalk::~CSCFakeDBCrosstalk()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbcrosstalk;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBCrosstalk::ReturnType
CSCFakeDBCrosstalk::produceDBCrosstalk(const CSCDBCrosstalkRcd& iRecord)
{
  return cndbcrosstalk;  
}

 void CSCFakeDBCrosstalk::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }



#ifndef _CSCFAKEDBCROSSTALK_H
#define _CSCFAKEDBCROSSTALK_H

#include <memory>
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCFakeDBCrosstalk: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeDBCrosstalk(const edm::ParameterSet&);
      ~CSCFakeDBCrosstalk() override;

      inline static CSCDBCrosstalk * prefillDBCrosstalk(); 

      typedef  std::shared_ptr<CSCDBCrosstalk> Pointer;

      Pointer produceDBCrosstalk(const CSCDBCrosstalkRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & ) override;
    Pointer cndbCrosstalk ;

};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCDBCrosstalk *  CSCFakeDBCrosstalk::prefillDBCrosstalk()
{
  int seed;
  float mean,min;
  int ii,jj,iii,jjj;
  const int MAX_SIZE = 217728; //or 252288 for ME4/2 chambers
  const int SLOPE_FACTOR=10000000;
  const int INTERCEPT_FACTOR=100000;
  
  CSCDBCrosstalk * cndbcrosstalk = new CSCDBCrosstalk();
  cndbcrosstalk->crosstalk.resize(MAX_SIZE);

  seed = 10000;	
  srand(seed);
  mean=-0.0009, min=0.035;
  ii=0,jj=0,iii=0,jjj=0;
  
  cndbcrosstalk->factor_slope = int (SLOPE_FACTOR);
  cndbcrosstalk->factor_intercept = int (INTERCEPT_FACTOR);
  
  for(int i=0; i<MAX_SIZE;i++){
    cndbcrosstalk->crosstalk[i].xtalk_slope_right = (short int) ((-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean)*SLOPE_FACTOR+0.5);
    cndbcrosstalk->crosstalk[i].xtalk_intercept_right= (short int) ((((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min)*INTERCEPT_FACTOR+0.5);
    cndbcrosstalk->crosstalk[i].xtalk_slope_left= (short int) ((-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean)*SLOPE_FACTOR+0.5);
    cndbcrosstalk->crosstalk[i].xtalk_intercept_left=(short int) ((((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min)*INTERCEPT_FACTOR+0.5); 
  
  
    //80 strips per chamber
    if(i<34561 && i%80==0){
      cndbcrosstalk->crosstalk[i].xtalk_slope_left=0;
      cndbcrosstalk->crosstalk[i].xtalk_intercept_left=0;
    }
    
    if(i!=0 && i<34561 && (i+1)%80==0){
      cndbcrosstalk->crosstalk[i].xtalk_slope_right=0;
      cndbcrosstalk->crosstalk[i].xtalk_intercept_right=0;
    }
    
    //64 strips per chamber
    if(i>34560 && i<48385 && i%64==0){
      cndbcrosstalk->crosstalk[i].xtalk_slope_left=0;
      cndbcrosstalk->crosstalk[i].xtalk_intercept_left=0;
    }
    
    if(i>34560 && i<48385 && (i+1)%64==0){
      cndbcrosstalk->crosstalk[i].xtalk_slope_right=0;
      cndbcrosstalk->crosstalk[i].xtalk_intercept_right=0;
    }
    
    //80 strips per chamber again
    if(i>48384 && i<143425){
      ii++;
      if(i>48384 && i<143425 && ii%80==0){
	cndbcrosstalk->crosstalk[i].xtalk_slope_left=0;
	cndbcrosstalk->crosstalk[i].xtalk_intercept_left=0;
      }
    }
    
    if(i>48384 && i<143425){
      jj++;
      if(i>48384 && i<143425 && (jj+1)%80==0){
	cndbcrosstalk->crosstalk[i].xtalk_slope_right=0;
	cndbcrosstalk->crosstalk[i].xtalk_intercept_right=0;
      }
    }
    
    //64 strips per chamber again
    if(i>143424 && i<157249 &&i%64==0){
      cndbcrosstalk->crosstalk[i].xtalk_slope_left=0;
      cndbcrosstalk->crosstalk[i].xtalk_intercept_left=0;
    }
    
    if(i>143424 && i<157249 && (i+1)%64==0){
      cndbcrosstalk->crosstalk[i].xtalk_slope_right=0;
      cndbcrosstalk->crosstalk[i].xtalk_intercept_right=0;
    }
    
    
    //80 strips per chamber last time
    if(i>157248){
      iii++;
      if(i>157248 && iii%80==0){
	cndbcrosstalk->crosstalk[i].xtalk_slope_left=0;
	cndbcrosstalk->crosstalk[i].xtalk_intercept_left=0;
      }
    }
    
    if(i>157248){
      jjj++;
      if(i>157248 && (jjj+1)%80==0){
	cndbcrosstalk->crosstalk[i].xtalk_slope_right=0;
	cndbcrosstalk->crosstalk[i].xtalk_intercept_right=0;
      }
    }
  }
  return cndbcrosstalk;
}  

#endif

  

 


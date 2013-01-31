#ifndef RecoLuminosity_LumiProducer_Lumi2DB_H 
#define RecoLuminosity_LumiProducer_Lumi2DB_H
#include "RelationalAccess/ConnectionService.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Blob.h"
#include "CoralBase/Exception.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "RecoLuminosity/LumiProducer/interface/LumiRawDataStructures.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RecoLuminosity/LumiProducer/interface/RevisionDML.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <map>
#include "TFile.h"
#include "TTree.h"
namespace lumi{
  class Lumi2DB : public DataPipe{
  public:
    const static unsigned int COMMITLSINTERVAL=500; //commit interval in LS,totalrow=nls*(1+nalgo)
    Lumi2DB(const std::string& dest);
    virtual unsigned long long retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~Lumi2DB();
  
    struct LumiSource{
      unsigned int run;
      unsigned int firstsection;
      char version[8];
      char datestr[9];
    };
    struct PerBXData{
      //int idx;
      float lumivalue;
      float lumierr;
      short lumiquality;
    };
    struct PerLumiData{
      float dtnorm;
      float lhcnorm;
      float instlumi;
      float instlumierror;
      short instlumiquality;
      short lumisectionquality;
      short cmsalive;
      std::string beammode;
      float beamenergy;
      short nlivebx;//how much is in the beamintensity vector
      short* bxindex;
      float* beamintensity_1;
      float* beamintensity_2;
      unsigned int cmslsnr;
      unsigned int lumilsnr;
      unsigned int startorbit;
      unsigned int numorbit;
      std::vector<PerBXData> bxET;
      std::vector<PerBXData> bxOCC1;
      std::vector<PerBXData> bxOCC2;
    };
    struct beamData{
      float energy;
      std::string mode;
      short nlivebx;
      short* bxindex;
      float* beamintensity_1;
      float* beamintensity_2;
    };
    typedef std::vector<PerLumiData> LumiResult;
    bool hasStableBeam( lumi::Lumi2DB::LumiResult::iterator lumiBeg,lumi::Lumi2DB::LumiResult::iterator lumiEnd );
  private:
    void parseSourceString(lumi::Lumi2DB::LumiSource& result)const;
    void retrieveBeamIntensity(HCAL_HLX::DIP_COMBINED_DATA* dataPtr, Lumi2DB::beamData&b)const;
    void writeAllLumiData(coral::ISessionProxy* session,unsigned int irunnumber,const std::string& ilumiversion,LumiResult::iterator lumiBeg,LumiResult::iterator lumiEnd);
    unsigned int writeAllLumiDataToSchema2(coral::ISessionProxy* session,const std::string& source,unsigned int runnumber,float bgev,unsigned int ncollidingbunches,LumiResult::iterator lumiBeg,LumiResult::iterator lumiEnd);
    void writeBeamIntensityOnly(coral::ISessionProxy* session,unsigned int irunnumber,const std::string& ilumiversion,LumiResult::iterator lumiBeg,LumiResult::iterator lumiEnd);
    bool isLumiDataValid(LumiResult::iterator lumiBeg,LumiResult::iterator lumiEnd);
    float applyCalibration(float varToCalibrate) const;
    void cleanTemporaryMemory( lumi::Lumi2DB::LumiResult::iterator lumiBeg,lumi::Lumi2DB::LumiResult::iterator lumiEnd);
  };//cl Lumi2DB
}//ns lumi

//
//implementation
//
float
lumi::Lumi2DB::applyCalibration(float varToCalibrate)const{ //#only used for writing into schema_v1
  return float(varToCalibrate)*m_norm;
}
bool
lumi::Lumi2DB::hasStableBeam( lumi::Lumi2DB::LumiResult::iterator lumiBeg,lumi::Lumi2DB::LumiResult::iterator lumiEnd ){
  //
  // the run has at least 1 stable beams LS
  //
  lumi::Lumi2DB::LumiResult::iterator lumiIt;  
  int nStable=0;
  for(lumiIt=lumiBeg;lumiIt!=lumiEnd;++lumiIt){
    if(lumiIt->beammode=="STABLE BEAMS"){
      ++nStable;
    }
  }
  if(nStable==0){
    return false;
  }
  return true;
}
bool
lumi::Lumi2DB::isLumiDataValid(lumi::Lumi2DB::LumiResult::iterator lumiBeg,lumi::Lumi2DB::LumiResult::iterator lumiEnd){
  //
  // validate lumidata: all ls has lumi less than 0.5e-08 before calibration, then invalid data
  //
  lumi::Lumi2DB::LumiResult::iterator lumiIt;
  int nBad=0;
  for(lumiIt=lumiBeg;lumiIt!=lumiEnd;++lumiIt){
    //std::cout<<"instlumi before calib "<<lumiIt->instlumi<<std::endl;
    if(lumiIt->instlumi<=0.5e-8){//cut before calib
      ++nBad;
    }
  }
  if(nBad==std::distance(lumiBeg,lumiEnd)){
    return false;
  }
  return true;
}
void
lumi::Lumi2DB::writeBeamIntensityOnly(
                            coral::ISessionProxy* session,
			    unsigned int irunnumber,
			    const std::string& ilumiversion,
                            lumi::Lumi2DB::LumiResult::iterator lumiBeg,
			    lumi::Lumi2DB::LumiResult::iterator lumiEnd
                            ){
  coral::AttributeList inputData;
  inputData.extend("bxindex",typeid(coral::Blob));
  inputData.extend("beamintensity_1",typeid(coral::Blob));
  inputData.extend("beamintensity_2",typeid(coral::Blob));
  inputData.extend("runnum",typeid(unsigned int));
  inputData.extend("startorbit",typeid(unsigned int));
  inputData.extend("lumiversion",typeid(std::string)); 
  coral::Blob& bxindex = inputData["bxindex"].data<coral::Blob>();
  coral::Blob& beamintensity_1 = inputData["beamintensity_1"].data<coral::Blob>();
  coral::Blob& beamintensity_2 = inputData["beamintensity_2"].data<coral::Blob>();
  unsigned int& runnumber = inputData["runnum"].data<unsigned int>();
  unsigned int& startorbit = inputData["startorbit"].data<unsigned int>();
  std::string& lumiversion = inputData["lumiversion"].data<std::string>();

  lumi::Lumi2DB::LumiResult::const_iterator lumiIt;
  coral::IBulkOperation* summaryUpdater=0;
  unsigned int totallumils=std::distance(lumiBeg,lumiEnd);
  unsigned int lumiindx=0;
  unsigned int comittedls=0;
  std::string setClause("CMSBXINDEXBLOB=:bxindex,BEAMINTENSITYBLOB_1=:beamintensity_1,BEAMINTENSITYBLOB_2=:beamintensity_2");
  std::string condition("RUNNUM=:runnum AND STARTORBIT=:startorbit AND LUMIVERSION=:lumiversion");
  runnumber=irunnumber;
  lumiversion=ilumiversion;
  for(lumiIt=lumiBeg;lumiIt!=lumiEnd;++lumiIt,++lumiindx){
    if(!session->transaction().isActive()){ 
      session->transaction().start(false);
    }
    startorbit=lumiIt->startorbit;
    //std::cout<<"runnumber "<<irunnumber<<" starorbit "<<startorbit<<" lumiversion "<<lumiversion<<" totallumils "<<totallumils<<std::endl;
    short nlivebx=lumiIt->nlivebx;
    if(nlivebx!=0){
      bxindex.resize(sizeof(short)*nlivebx);
      beamintensity_1.resize(sizeof(float)*nlivebx);
      beamintensity_2.resize(sizeof(float)*nlivebx);
      void* bxindex_StartAddress = bxindex.startingAddress();      
      void* beamIntensity1_StartAddress = beamintensity_1.startingAddress();
      void* beamIntensity2_StartAddress = beamintensity_2.startingAddress();
      std::memmove(bxindex_StartAddress,lumiIt->bxindex,sizeof(short)*nlivebx);
      std::memmove(beamIntensity1_StartAddress,lumiIt->beamintensity_1,sizeof(float)*nlivebx);
      std::memmove(beamIntensity2_StartAddress,lumiIt->beamintensity_2,sizeof(float)*nlivebx);
      ::free(lumiIt->bxindex);
      ::free(lumiIt->beamintensity_1);
      ::free(lumiIt->beamintensity_2);
    }else{
      bxindex.resize(0);
      beamintensity_1.resize(0);
      beamintensity_2.resize(0);
    }
    coral::ITable& summarytable=session->nominalSchema().tableHandle(LumiNames::lumisummaryTableName());
    summaryUpdater=summarytable.dataEditor().bulkUpdateRows(setClause,condition,inputData,totallumils);
    summaryUpdater->processNextIteration();
    summaryUpdater->flush();
    ++comittedls;
    if(comittedls==Lumi2DB::COMMITLSINTERVAL){
      std::cout<<"\t committing in LS chunck "<<comittedls<<std::endl; 
      delete summaryUpdater;
      summaryUpdater=0;
      session->transaction().commit();
      comittedls=0;
      std::cout<<"\t committed "<<std::endl; 
    }else if( lumiindx==(totallumils-1) ){
      std::cout<<"\t committing at the end"<<std::endl; 
      delete summaryUpdater; summaryUpdater=0;
      session->transaction().commit();
      std::cout<<"\t done"<<std::endl; 
    }
  }
}
void
lumi::Lumi2DB::writeAllLumiData(
			    coral::ISessionProxy* session,
			    unsigned int irunnumber,
			    const std::string& ilumiversion,
			    lumi::Lumi2DB::LumiResult::iterator lumiBeg,
			    lumi::Lumi2DB::LumiResult::iterator lumiEnd	){
  coral::AttributeList summaryData;
  coral::AttributeList detailData;
  summaryData.extend("LUMISUMMARY_ID",typeid(unsigned long long));
  summaryData.extend("RUNNUM",typeid(unsigned int));
  summaryData.extend("CMSLSNUM",typeid(unsigned int));
  summaryData.extend("LUMILSNUM",typeid(unsigned int));
  summaryData.extend("LUMIVERSION",typeid(std::string));
  summaryData.extend("DTNORM",typeid(float));
  summaryData.extend("LHCNORM",typeid(float));
  summaryData.extend("INSTLUMI",typeid(float));
  summaryData.extend("INSTLUMIERROR",typeid(float));
  summaryData.extend("INSTLUMIQUALITY",typeid(short));
  summaryData.extend("LUMISECTIONQUALITY",typeid(short));
  summaryData.extend("CMSALIVE",typeid(short));
  summaryData.extend("STARTORBIT",typeid(unsigned int));
  summaryData.extend("NUMORBIT",typeid(unsigned int));
  summaryData.extend("BEAMENERGY",typeid(float));
  summaryData.extend("BEAMSTATUS",typeid(std::string));
  summaryData.extend("CMSBXINDEXBLOB",typeid(coral::Blob));
  summaryData.extend("BEAMINTENSITYBLOB_1",typeid(coral::Blob));
  summaryData.extend("BEAMINTENSITYBLOB_2",typeid(coral::Blob));
  
  detailData.extend("LUMIDETAIL_ID",typeid(unsigned long long));
  detailData.extend("LUMISUMMARY_ID",typeid(unsigned long long));
  detailData.extend("BXLUMIVALUE",typeid(coral::Blob));
  detailData.extend("BXLUMIERROR",typeid(coral::Blob));
  detailData.extend("BXLUMIQUALITY",typeid(coral::Blob));
  detailData.extend("ALGONAME",typeid(std::string));
  
  unsigned long long& lumisummary_id=summaryData["LUMISUMMARY_ID"].data<unsigned long long>();
  unsigned int& lumirunnum = summaryData["RUNNUM"].data<unsigned int>();
  std::string& lumiversion = summaryData["LUMIVERSION"].data<std::string>();
  float& dtnorm = summaryData["DTNORM"].data<float>();
  float& lhcnorm = summaryData["LHCNORM"].data<float>();
  float& instlumi = summaryData["INSTLUMI"].data<float>();
  float& instlumierror = summaryData["INSTLUMIERROR"].data<float>();
  short& instlumiquality = summaryData["INSTLUMIQUALITY"].data<short>();
  short& lumisectionquality = summaryData["LUMISECTIONQUALITY"].data<short>();
  short& alive = summaryData["CMSALIVE"].data<short>();
  unsigned int& lumilsnr = summaryData["LUMILSNUM"].data<unsigned int>();
  unsigned int& cmslsnr = summaryData["CMSLSNUM"].data<unsigned int>();
  unsigned int& startorbit = summaryData["STARTORBIT"].data<unsigned int>();
  unsigned int& numorbit = summaryData["NUMORBIT"].data<unsigned int>();
  float& beamenergy = summaryData["BEAMENERGY"].data<float>();
  std::string& beamstatus = summaryData["BEAMSTATUS"].data<std::string>();
  coral::Blob& bxindex = summaryData["CMSBXINDEXBLOB"].data<coral::Blob>();
  coral::Blob& beamintensity_1 = summaryData["BEAMINTENSITYBLOB_1"].data<coral::Blob>();
  coral::Blob& beamintensity_2 = summaryData["BEAMINTENSITYBLOB_2"].data<coral::Blob>();
  
  unsigned long long& lumidetail_id=detailData["LUMIDETAIL_ID"].data<unsigned long long>();
  unsigned long long& d2lumisummary_id=detailData["LUMISUMMARY_ID"].data<unsigned long long>();
  coral::Blob& bxlumivalue=detailData["BXLUMIVALUE"].data<coral::Blob>();
  coral::Blob& bxlumierror=detailData["BXLUMIERROR"].data<coral::Blob>();
  coral::Blob& bxlumiquality=detailData["BXLUMIQUALITY"].data<coral::Blob>();
  std::string& algoname=detailData["ALGONAME"].data<std::string>();
    
  lumi::Lumi2DB::LumiResult::const_iterator lumiIt;
  coral::IBulkOperation* summaryInserter=0;
  coral::IBulkOperation* detailInserter=0;
  //one loop for ids
  //nested transaction doesn't work with bulk inserter
  unsigned int totallumils=std::distance(lumiBeg,lumiEnd);
  unsigned int lumiindx=0;
  std::map< unsigned long long,std::vector<unsigned long long> > idallocationtable;
  std::cout<<"\t allocating total lumisummary ids "<<totallumils<<std::endl; 
  std::cout<<"\t allocating total lumidetail ids "<<totallumils*lumi::N_LUMIALGO<<std::endl; 

  session->transaction().start(false);
  lumi::idDealer idg(session->nominalSchema());
  unsigned long long lumisummaryID = idg.generateNextIDForTable(LumiNames::lumisummaryTableName(),totallumils)-totallumils;
  unsigned long long lumidetailID=idg.generateNextIDForTable(LumiNames::lumidetailTableName(),totallumils*lumi::N_LUMIALGO)-totallumils*lumi::N_LUMIALGO;
  session->transaction().commit();
  for(lumiIt=lumiBeg;lumiIt!=lumiEnd;++lumiIt,++lumiindx,++lumisummaryID){
    std::vector< unsigned long long > allIDs;
    allIDs.reserve(1+lumi::N_LUMIALGO);
    allIDs.push_back(lumisummaryID);
    for( unsigned int j=0; j<lumi::N_LUMIALGO; ++j, ++lumidetailID){
      allIDs.push_back(lumidetailID);
    }
    idallocationtable.insert(std::make_pair(lumiindx,allIDs));
  }
  std::cout<<"\t all ids allocated"<<std::endl; 
  lumiindx=0;
  unsigned int comittedls=0;
  for(lumiIt=lumiBeg;lumiIt!=lumiEnd;++lumiIt,++lumiindx){
    if(!session->transaction().isActive()){ 
      session->transaction().start(false);
      coral::ITable& summarytable=session->nominalSchema().tableHandle(LumiNames::lumisummaryTableName());
      summaryInserter=summarytable.dataEditor().bulkInsert(summaryData,totallumils);
      coral::ITable& detailtable=session->nominalSchema().tableHandle(LumiNames::lumidetailTableName());
      detailInserter=detailtable.dataEditor().bulkInsert(detailData,totallumils*lumi::N_LUMIALGO);    
    }
    lumisummary_id=idallocationtable[lumiindx][0];
    lumirunnum = irunnumber;
    lumiversion = ilumiversion;
    dtnorm = lumiIt->dtnorm;
    lhcnorm = lumiIt->lhcnorm;
    //instlumi = lumiIt->instlumi;
    //instlumierror = lumiIt->instlumierror;
    instlumi = applyCalibration(lumiIt->instlumi);
    instlumierror = applyCalibration(lumiIt->instlumierror);
    instlumiquality = lumiIt->instlumiquality;
    lumisectionquality = lumiIt->lumisectionquality;
    alive = lumiIt->cmsalive;
    cmslsnr = lumiIt->cmslsnr;
      
    lumilsnr = lumiIt->lumilsnr;
    startorbit = lumiIt->startorbit;
    numorbit = lumiIt->numorbit;
    beamenergy = lumiIt->beamenergy;
    beamstatus = lumiIt->beammode;
    short nlivebx=lumiIt->nlivebx;
    //std::cout<<"nlivebx "<<nlivebx<<std::endl;
    if(nlivebx!=0){
      bxindex.resize(sizeof(short)*nlivebx);
      beamintensity_1.resize(sizeof(float)*nlivebx);
      beamintensity_2.resize(sizeof(float)*nlivebx);
      void* bxindex_StartAddress = bxindex.startingAddress();      
      void* beamIntensity1_StartAddress = beamintensity_1.startingAddress();
      void* beamIntensity2_StartAddress = beamintensity_2.startingAddress();
      std::memmove(bxindex_StartAddress,lumiIt->bxindex,sizeof(short)*nlivebx);
      std::memmove(beamIntensity1_StartAddress,lumiIt->beamintensity_1,sizeof(float)*nlivebx);
      std::memmove(beamIntensity2_StartAddress,lumiIt->beamintensity_2,sizeof(float)*nlivebx);
      //::free(lumiIt->bxindex);
      //::free(lumiIt->beamintensity_1);
      //::free(lumiIt->beamintensity_2);
    }else{
      bxindex.resize(0);
      beamintensity_1.resize(0);
      beamintensity_2.resize(0);
    }
    //insert the new row
    summaryInserter->processNextIteration();
    summaryInserter->flush();
    unsigned int algoindx=1;
    for( unsigned int j=0; j<lumi:: N_LUMIALGO; ++j,++algoindx ){
      d2lumisummary_id=idallocationtable[lumiindx].at(0);
      lumidetail_id=idallocationtable[lumiindx].at(algoindx);
      std::vector<PerBXData>::const_iterator bxIt;
      std::vector<PerBXData>::const_iterator bxBeg;
      std::vector<PerBXData>::const_iterator bxEnd;
      if(j==0) {
	algoname=std::string("ET");
	bxBeg=lumiIt->bxET.begin();
	bxEnd=lumiIt->bxET.end();
      }
      if(j==1) {
	algoname=std::string("OCC1");
	bxBeg=lumiIt->bxOCC1.begin();
	bxEnd=lumiIt->bxOCC1.end();
      }
      if(j==2) {
	algoname=std::string("OCC2");
	bxBeg=lumiIt->bxOCC2.begin();
	bxEnd=lumiIt->bxOCC2.end();
      }
      float lumivalue[lumi::N_BX]={0.0};
      float lumierror[lumi::N_BX]={0.0};
      int lumiquality[lumi::N_BX]={0};
      bxlumivalue.resize(sizeof(float)*lumi::N_BX);
      bxlumierror.resize(sizeof(float)*lumi::N_BX);
      bxlumiquality.resize(sizeof(short)*lumi::N_BX);
      void* bxlumivalueStartAddress=bxlumivalue.startingAddress();
      void* bxlumierrorStartAddress=bxlumierror.startingAddress();
      void* bxlumiqualityStartAddress=bxlumiquality.startingAddress();
      unsigned int k=0;
      for( bxIt=bxBeg;bxIt!=bxEnd;++bxIt,++k  ){	    
	lumivalue[k]=bxIt->lumivalue;
	lumierror[k]=bxIt->lumierr;
	lumiquality[k]=bxIt->lumiquality;
      }
      std::memmove(bxlumivalueStartAddress,lumivalue,sizeof(float)*lumi::N_BX);
      std::memmove(bxlumierrorStartAddress,lumierror,sizeof(float)*lumi::N_BX);
      std::memmove(bxlumiqualityStartAddress,lumiquality,sizeof(short)*lumi::N_BX);
      detailInserter->processNextIteration();
    }
    detailInserter->flush();
    ++comittedls;
    if(comittedls==Lumi2DB::COMMITLSINTERVAL){
      std::cout<<"\t committing in LS chunck "<<comittedls<<std::endl; 
      delete summaryInserter;
      summaryInserter=0;
      delete detailInserter;
      detailInserter=0;
      session->transaction().commit();
      comittedls=0;
      std::cout<<"\t committed "<<std::endl; 
    }else if( lumiindx==(totallumils-1) ){
      std::cout<<"\t committing at the end"<<std::endl; 
      delete summaryInserter; summaryInserter=0;
      delete detailInserter; detailInserter=0;
      session->transaction().commit();
      std::cout<<"\t done"<<std::endl; 
    }
  }
}

unsigned int
lumi::Lumi2DB::writeAllLumiDataToSchema2(
			    coral::ISessionProxy* session,
			    const std::string& source,
			    unsigned int irunnumber,
			    float bgev,
			    unsigned int ncollidingbunches,
			    lumi::Lumi2DB::LumiResult::iterator lumiBeg,
			    lumi::Lumi2DB::LumiResult::iterator lumiEnd	){
  ///
  //output: lumi data id
  ///
  std::cout<<"writeAllLumiDataToSchema2"<<std::endl;
  coral::AttributeList summaryData;
  summaryData.extend("DATA_ID",typeid(unsigned long long));
  summaryData.extend("RUNNUM",typeid(unsigned int));
  summaryData.extend("LUMILSNUM",typeid(unsigned int));
  summaryData.extend("CMSLSNUM",typeid(unsigned int));
  summaryData.extend("INSTLUMI",typeid(float));
  summaryData.extend("INSTLUMIERROR",typeid(float));
  summaryData.extend("INSTLUMIQUALITY",typeid(short));
  summaryData.extend("BEAMSTATUS",typeid(std::string));
  summaryData.extend("BEAMENERGY",typeid(float));
  summaryData.extend("NUMORBIT",typeid(unsigned int));
  summaryData.extend("STARTORBIT",typeid(unsigned int));
  summaryData.extend("CMSBXINDEXBLOB",typeid(coral::Blob));
  summaryData.extend("BEAMINTENSITYBLOB_1",typeid(coral::Blob));
  summaryData.extend("BEAMINTENSITYBLOB_2",typeid(coral::Blob));
  summaryData.extend("BXLUMIVALUE_OCC1",typeid(coral::Blob));
  summaryData.extend("BXLUMIERROR_OCC1",typeid(coral::Blob));
  summaryData.extend("BXLUMIQUALITY_OCC1",typeid(coral::Blob));
  summaryData.extend("BXLUMIVALUE_OCC2",typeid(coral::Blob));
  summaryData.extend("BXLUMIERROR_OCC2",typeid(coral::Blob));
  summaryData.extend("BXLUMIQUALITY_OCC2",typeid(coral::Blob));
  summaryData.extend("BXLUMIVALUE_ET",typeid(coral::Blob));
  summaryData.extend("BXLUMIERROR_ET",typeid(coral::Blob));
  summaryData.extend("BXLUMIQUALITY_ET",typeid(coral::Blob));

  unsigned long long& data_id=summaryData["DATA_ID"].data<unsigned long long>();
  unsigned int& lumirunnum = summaryData["RUNNUM"].data<unsigned int>();
  unsigned int& lumilsnr = summaryData["LUMILSNUM"].data<unsigned int>();
  unsigned int& cmslsnr = summaryData["CMSLSNUM"].data<unsigned int>();
  float& instlumi = summaryData["INSTLUMI"].data<float>();
  float& instlumierror = summaryData["INSTLUMIERROR"].data<float>();
  short& instlumiquality = summaryData["INSTLUMIQUALITY"].data<short>();
  std::string& beamstatus = summaryData["BEAMSTATUS"].data<std::string>();
  float& beamenergy = summaryData["BEAMENERGY"].data<float>(); 
  unsigned int& numorbit = summaryData["NUMORBIT"].data<unsigned int>();
  unsigned int& startorbit = summaryData["STARTORBIT"].data<unsigned int>();
  coral::Blob& bxindex = summaryData["CMSBXINDEXBLOB"].data<coral::Blob>();
  coral::Blob& beamintensity_1 = summaryData["BEAMINTENSITYBLOB_1"].data<coral::Blob>();
  coral::Blob& beamintensity_2 = summaryData["BEAMINTENSITYBLOB_2"].data<coral::Blob>();  
  coral::Blob& bxlumivalue_et=summaryData["BXLUMIVALUE_ET"].data<coral::Blob>();
  coral::Blob& bxlumierror_et=summaryData["BXLUMIERROR_ET"].data<coral::Blob>();
  coral::Blob& bxlumiquality_et=summaryData["BXLUMIQUALITY_ET"].data<coral::Blob>();
  coral::Blob& bxlumivalue_occ1=summaryData["BXLUMIVALUE_OCC1"].data<coral::Blob>();
  coral::Blob& bxlumierror_occ1=summaryData["BXLUMIERROR_OCC1"].data<coral::Blob>();
  coral::Blob& bxlumiquality_occ1=summaryData["BXLUMIQUALITY_OCC1"].data<coral::Blob>();
  coral::Blob& bxlumivalue_occ2=summaryData["BXLUMIVALUE_OCC2"].data<coral::Blob>();
  coral::Blob& bxlumierror_occ2=summaryData["BXLUMIERROR_OCC2"].data<coral::Blob>();
  coral::Blob& bxlumiquality_occ2=summaryData["BXLUMIQUALITY_OCC2"].data<coral::Blob>();

  lumi::Lumi2DB::LumiResult::const_iterator lumiIt;
  coral::IBulkOperation* summaryInserter=0;

  unsigned int totallumils=std::distance(lumiBeg,lumiEnd);
  unsigned int lumiindx=0;
  unsigned int comittedls=0;
  
  unsigned long long branch_id=3;
  std::string branch_name("DATA");
  lumi::RevisionDML revisionDML;
  lumi::RevisionDML::LumiEntry lumirundata;
  std::stringstream op;
  op<<irunnumber;
  std::string runnumberStr=op.str();
  session->transaction().start(false);
  lumirundata.entry_name=runnumberStr;
  lumirundata.source=source;
  lumirundata.runnumber=irunnumber;
  lumirundata.bgev=bgev;
  lumirundata.ncollidingbunches=ncollidingbunches;
  lumirundata.data_id=0;
  lumirundata.entry_id=revisionDML.getEntryInBranchByName(session->nominalSchema(),lumi::LumiNames::lumidataTableName(),runnumberStr,branch_name);
  //std::cout<<"entry_id "<<lumirundata.entry_id<<std::endl;
  if(lumirundata.entry_id==0){
    revisionDML.bookNewEntry(session->nominalSchema(),LumiNames::lumidataTableName(),lumirundata);
    std::cout<<"it's a new run lumirundata revision_id "<<lumirundata.revision_id<<" entry_id "<<lumirundata.entry_id<<" data_id "<<lumirundata.data_id<<std::endl;
    revisionDML.addEntry(session->nominalSchema(),LumiNames::lumidataTableName(),lumirundata,branch_id,branch_name);
    std::cout<<"added entry "<<std::endl;
  }else{
    revisionDML.bookNewRevision(session->nominalSchema(),LumiNames::lumidataTableName(),lumirundata);
    std::cout<<"lumirundata revision_id "<<lumirundata.revision_id<<" entry_id "<<lumirundata.entry_id<<" data_id "<<lumirundata.data_id<<std::endl;
    revisionDML.addRevision(session->nominalSchema(),LumiNames::lumidataTableName(),lumirundata,branch_id,branch_name);
  }
  revisionDML.insertLumiRunData(session->nominalSchema(),lumirundata);
  for(lumiIt=lumiBeg;lumiIt!=lumiEnd;++lumiIt,++lumiindx){
    if(!session->transaction().isActive()){ 
      session->transaction().start(false);
      coral::ITable& summarytable=session->nominalSchema().tableHandle(LumiNames::lumisummaryv2TableName());
      summaryInserter=summarytable.dataEditor().bulkInsert(summaryData,totallumils);
    }else{
      if(lumiIt==lumiBeg){
	coral::ITable& summarytable=session->nominalSchema().tableHandle(LumiNames::lumisummaryv2TableName());
	summaryInserter=summarytable.dataEditor().bulkInsert(summaryData,totallumils);
      }
    }
    data_id = lumirundata.data_id;
    lumirunnum = irunnumber;
    lumilsnr = lumiIt->lumilsnr;
    cmslsnr = lumiIt->cmslsnr;
    instlumi = lumiIt->instlumi; // not calibrated!
    instlumierror = lumiIt->instlumierror; // not calibrated!
    instlumiquality = lumiIt->instlumiquality;
    beamstatus = lumiIt->beammode;  
    beamenergy = lumiIt->beamenergy;
    numorbit = lumiIt->numorbit;
    startorbit = lumiIt->startorbit;
    short nlivebx=lumiIt->nlivebx;
    //std::cout<<"nlivebx "<<nlivebx<<std::endl;
    if(nlivebx!=0){
      bxindex.resize(sizeof(short)*nlivebx);
      beamintensity_1.resize(sizeof(float)*nlivebx);
      beamintensity_2.resize(sizeof(float)*nlivebx);
      void* bxindex_StartAddress = bxindex.startingAddress();      
      void* beamIntensity1_StartAddress = beamintensity_1.startingAddress();
      void* beamIntensity2_StartAddress = beamintensity_2.startingAddress();
      std::memmove(bxindex_StartAddress,lumiIt->bxindex,sizeof(short)*nlivebx);
      std::memmove(beamIntensity1_StartAddress,lumiIt->beamintensity_1,sizeof(float)*nlivebx);
      std::memmove(beamIntensity2_StartAddress,lumiIt->beamintensity_2,sizeof(float)*nlivebx);
      //::free(lumiIt->bxindex);
      //::free(lumiIt->beamintensity_1);
      //::free(lumiIt->beamintensity_2);
    }else{
      bxindex.resize(0);
      beamintensity_1.resize(0);
      beamintensity_2.resize(0);
    }    
    for( unsigned int j=0; j<lumi:: N_LUMIALGO; ++j ){
      std::vector<PerBXData>::const_iterator bxIt;
      std::vector<PerBXData>::const_iterator bxBeg;
      std::vector<PerBXData>::const_iterator bxEnd;
      if(j==0) {//the push_back order in the input data is ET,OCC1,OCC2
	//algoname=std::string("ET");
	bxBeg=lumiIt->bxET.begin();
	bxEnd=lumiIt->bxET.end();
	float lumivalue[lumi::N_BX]={0.0};
	float lumierror[lumi::N_BX]={0.0};
	int lumiquality[lumi::N_BX]={0};
	bxlumivalue_et.resize(sizeof(float)*lumi::N_BX);
	bxlumierror_et.resize(sizeof(float)*lumi::N_BX);
	bxlumiquality_et.resize(sizeof(short)*lumi::N_BX);
	void* bxlumivalueStartAddress=bxlumivalue_et.startingAddress();
	void* bxlumierrorStartAddress=bxlumierror_et.startingAddress();
	void* bxlumiqualityStartAddress=bxlumiquality_et.startingAddress();
	unsigned int k=0;
	for( bxIt=bxBeg;bxIt!=bxEnd;++bxIt,++k  ){	    
	  lumivalue[k]=bxIt->lumivalue;
	  lumierror[k]=bxIt->lumierr;
	  lumiquality[k]=bxIt->lumiquality;
	}
	std::memmove(bxlumivalueStartAddress,lumivalue,sizeof(float)*lumi::N_BX);
	std::memmove(bxlumierrorStartAddress,lumierror,sizeof(float)*lumi::N_BX);
	std::memmove(bxlumiqualityStartAddress,lumiquality,sizeof(short)*lumi::N_BX);
      }
      if(j==1) {
	//algoname=std::string("OCC1");
	bxBeg=lumiIt->bxOCC1.begin();
	bxEnd=lumiIt->bxOCC1.end();
	float lumivalue[lumi::N_BX]={0.0};
	float lumierror[lumi::N_BX]={0.0};
	int lumiquality[lumi::N_BX]={0};
	bxlumivalue_occ1.resize(sizeof(float)*lumi::N_BX);
	bxlumierror_occ1.resize(sizeof(float)*lumi::N_BX);
	bxlumiquality_occ1.resize(sizeof(short)*lumi::N_BX);
	void* bxlumivalueStartAddress=bxlumivalue_occ1.startingAddress();
	void* bxlumierrorStartAddress=bxlumierror_occ1.startingAddress();
	void* bxlumiqualityStartAddress=bxlumiquality_occ1.startingAddress();
	unsigned int k=0;
	for( bxIt=bxBeg;bxIt!=bxEnd;++bxIt,++k  ){	    
	  lumivalue[k]=bxIt->lumivalue;
	  lumierror[k]=bxIt->lumierr;
	  lumiquality[k]=bxIt->lumiquality;
	}
	std::memmove(bxlumivalueStartAddress,lumivalue,sizeof(float)*lumi::N_BX);
	std::memmove(bxlumierrorStartAddress,lumierror,sizeof(float)*lumi::N_BX);
	std::memmove(bxlumiqualityStartAddress,lumiquality,sizeof(short)*lumi::N_BX);
      }
      if(j==2) {
	//algoname=std::string("OCC2");
	bxBeg=lumiIt->bxOCC2.begin();
	bxEnd=lumiIt->bxOCC2.end();
	float lumivalue[lumi::N_BX]={0.0};
	float lumierror[lumi::N_BX]={0.0};
	int lumiquality[lumi::N_BX]={0};
	bxlumivalue_occ2.resize(sizeof(float)*lumi::N_BX);
	bxlumierror_occ2.resize(sizeof(float)*lumi::N_BX);
	bxlumiquality_occ2.resize(sizeof(short)*lumi::N_BX);
	void* bxlumivalueStartAddress=bxlumivalue_occ2.startingAddress();
	void* bxlumierrorStartAddress=bxlumierror_occ2.startingAddress();
	void* bxlumiqualityStartAddress=bxlumiquality_occ2.startingAddress();
	unsigned int k=0;
	for( bxIt=bxBeg;bxIt!=bxEnd;++bxIt,++k  ){	    
	  lumivalue[k]=bxIt->lumivalue;
	  lumierror[k]=bxIt->lumierr;
	  lumiquality[k]=bxIt->lumiquality;
	}
	std::memmove(bxlumivalueStartAddress,lumivalue,sizeof(float)*lumi::N_BX);
	std::memmove(bxlumierrorStartAddress,lumierror,sizeof(float)*lumi::N_BX);
	std::memmove(bxlumiqualityStartAddress,lumiquality,sizeof(short)*lumi::N_BX);
      }
    }
    summaryInserter->processNextIteration();
    summaryInserter->flush();
    ++comittedls;
    if(comittedls==Lumi2DB::COMMITLSINTERVAL){
      std::cout<<"\t committing in LS chunck "<<comittedls<<std::endl; 
      delete summaryInserter;
      summaryInserter=0;
      session->transaction().commit();
      comittedls=0;
      std::cout<<"\t committed "<<std::endl; 
    }else if( lumiindx==(totallumils-1) ){
      std::cout<<"\t committing at the end"<<std::endl; 
      delete summaryInserter; summaryInserter=0;
      session->transaction().commit();
      std::cout<<"\t done"<<std::endl; 
    }
  }
  return lumirundata.data_id;
}

void lumi::Lumi2DB::cleanTemporaryMemory( lumi::Lumi2DB::LumiResult::iterator lumiBeg,
					  lumi::Lumi2DB::LumiResult::iterator lumiEnd){
  lumi::Lumi2DB::LumiResult::const_iterator lumiIt;
  for(lumiIt=lumiBeg;lumiIt!=lumiEnd;++lumiIt){
    ::free(lumiIt->bxindex);
    ::free(lumiIt->beamintensity_1);
    ::free(lumiIt->beamintensity_2);
  }
  
}
lumi::Lumi2DB::Lumi2DB(const std::string& dest):DataPipe(dest){}
void 
lumi::Lumi2DB::parseSourceString(lumi::Lumi2DB::LumiSource& result)const{
  //parse lumi source file name
  if(m_source.length()==0) throw lumi::Exception("lumi source is not set","parseSourceString","Lumi2DB");
  //std::cout<<"source "<<m_source<<std::endl;
  size_t tempIndex = m_source.find_last_of(".");
  size_t nameLength = m_source.length();
  std::string FileSuffix= m_source.substr(tempIndex+1,nameLength - tempIndex);
  std::string::size_type lastPos=m_source.find_first_not_of("_",0);
  std::string::size_type pos = m_source.find_first_of("_",lastPos);
  std::vector<std::string> pieces;
  while( std::string::npos != pos || std::string::npos != lastPos){
    pieces.push_back(m_source.substr(lastPos,pos-lastPos));
    lastPos=m_source.find_first_not_of("_",pos);
    pos=m_source.find_first_of("_",lastPos);
  }
  if( pieces[1]!="LUMI" || pieces[2]!="RAW" || FileSuffix!="root"){
    throw lumi::Exception("not lumi raw data file CMS_LUMI_RAW","parseSourceString","Lumi2DB");
  }
  std::strcpy(result.datestr,pieces[3].c_str());
  std::strcpy(result.version,pieces[5].c_str());
  //std::cout<<"version : "<<result.version<<std::endl;
  result.run = atoi(pieces[4].c_str());
  //std::cout<<"run : "<<result.run<<std::endl;
  result.firstsection = atoi(pieces[5].c_str());
  //std::cout<<"first section : "<< result.firstsection<<std::endl;
}

void
lumi::Lumi2DB::retrieveBeamIntensity(HCAL_HLX::DIP_COMBINED_DATA* dataPtr, Lumi2DB::beamData&b)const{
   if(dataPtr==0){
      std::cout<<"HCAL_HLX::DIP_COMBINED_DATA* dataPtr=0"<<std::endl;
      b.bxindex=0;
      b.beamintensity_1=0;
      b.beamintensity_2=0;
      b.nlivebx=0;
   }else{
      b.bxindex=(short*)::malloc(sizeof(short)*lumi::N_BX);
      b.beamintensity_1=(float*)::malloc(sizeof(float)*lumi::N_BX);
      b.beamintensity_2=(float*)::malloc(sizeof(float)*lumi::N_BX);
      
      short a=0;//a is position in lumidetail array
      for(unsigned int i=0;i<lumi::N_BX;++i){
	 if( i==0 ){
	    if(dataPtr->Beam[0].averageBunchIntensities[0]>0 || dataPtr->Beam[1].averageBunchIntensities[0]>0 ){
	       b.bxindex[a]=0;
	       b.beamintensity_1[a]=dataPtr->Beam[0].averageBunchIntensities[0];
	       b.beamintensity_2[a]=dataPtr->Beam[1].averageBunchIntensities[0];
	       ++a;
	    }
	    continue;
	 }
	 if(dataPtr->Beam[0].averageBunchIntensities[i-1]>0 || dataPtr->Beam[1].averageBunchIntensities[i-1]>0){
	    b.bxindex[a]=i;
	    b.beamintensity_1[a]=dataPtr->Beam[0].averageBunchIntensities[i-1];
	    b.beamintensity_2[a]=dataPtr->Beam[1].averageBunchIntensities[i-1];
	    ++a;
	    //if(i!=0){
	    // std::cout<<"beam intensity "<<dataPtr->sectionNumber<<" "<<dataPtr->timestamp-1262300400<<" "<<(i-1)*10+1<<" "<<b.beamintensity_1[a]<<" "<<b.beamintensity_2[a]<<std::endl;
	    //}
	 }
      }
      b.nlivebx=a;
   }
}
/**
   retrieve lumi per ls data from root file
 **/
unsigned long long
lumi::Lumi2DB::retrieveData( unsigned int runnumber){
  lumi::Lumi2DB::LumiResult lumiresult;
  //check filename is in  lumiraw format
  lumi::Lumi2DB::LumiSource filenamecontent;
  unsigned int lumidataid=0;
  try{
    parseSourceString(filenamecontent);
  }catch(const lumi::Exception& er){
    std::cout<<er.what()<<std::endl;
    throw er;
  }
  if( filenamecontent.run!=runnumber ){
    throw lumi::Exception("runnumber in file name does not match requested run number","retrieveData","Lumi2DB");
  }
  TFile* source=TFile::Open(m_source.c_str(),"READ");
  TTree *hlxtree = (TTree*)source->Get("HLXData");
  if(!hlxtree){
    throw lumi::Exception(std::string("non-existing HLXData "),"retrieveData","Lumi2DB");
  }
  //hlxtree->Print();
  std::auto_ptr<HCAL_HLX::LUMI_SECTION> localSection(new HCAL_HLX::LUMI_SECTION);
  HCAL_HLX::LUMI_SECTION_HEADER* lumiheader = &(localSection->hdr);
  HCAL_HLX::LUMI_SUMMARY* lumisummary = &(localSection->lumiSummary);
  HCAL_HLX::LUMI_DETAIL* lumidetail = &(localSection->lumiDetail);
  
  hlxtree->SetBranchAddress("Header.",&lumiheader);
  hlxtree->SetBranchAddress("Summary.",&lumisummary);
  hlxtree->SetBranchAddress("Detail.",&lumidetail);
   
  size_t nentries=hlxtree->GetEntries();
  unsigned int nstablebeam=0;
  float bgev=0.0;
  //source->GetListOfKeys()->Print();
  std::map<unsigned int, Lumi2DB::beamData> dipmap;
  TTree *diptree= (TTree*)source->Get("DIPCombined");
  if(diptree){
    //throw lumi::Exception(std::string("non-existing DIPData "),"retrieveData","Lumi2DB");
    std::auto_ptr<HCAL_HLX::DIP_COMBINED_DATA> dipdata(new HCAL_HLX::DIP_COMBINED_DATA);
    diptree->SetBranchAddress("DIPCombined.",&dipdata);
    size_t ndipentries=diptree->GetEntries();
    for(size_t i=0;i<ndipentries;++i){
      diptree->GetEntry(i);
      //unsigned int fillnumber=dipdata->FillNumber;
      //std::vector<short> collidingidx;collidingidx.reserve(LUMI::N_BX);
      //for(unsigned int i=0;i<lumi::N_BX;++i){
      //int isb1colliding=dipdata->beam[0].beamConfig[i];
      //int isb2colliding=dipdata->beam[1].beamConfig[i];
      //if(isb1colliding && isb2colliding&&isb1colliding==1&&isb2colliding==1){
      //  collidingidx.push_back(i);
      //	}
      //}
      beamData b;
      //std::cout<<"Beam Mode : "<<dipdata->beamMode<<"\n";
      //std::cout<<"Beam Energy : "<<dipdata->Energy<<"\n";
      //std::cout<<"sectionUmber : "<<dipdata->sectionNumber<<"\n";
      unsigned int dipls=dipdata->sectionNumber;
      if (std::string(dipdata->beamMode).empty()){
	b.mode="N/A";
      }else{
	b.mode=dipdata->beamMode;
      }
      b.energy=dipdata->Energy;
      if(b.mode=="STABLE BEAMS"){
	++nstablebeam;
	bgev+=b.energy;
      }
      this->retrieveBeamIntensity(dipdata.get(),b);
      dipmap.insert(std::make_pair(dipls,b));
    }
  }else{
    for(size_t i=0;i<nentries;++i){
      beamData b;
      b.mode="N/A";
      b.energy=0.0;
      this->retrieveBeamIntensity(0,b);
      dipmap.insert(std::make_pair(i,b));
    }
  }
  //diptree->Print();
 
  size_t ncmslumi=0;
  std::cout<<"processing total lumi lumisection "<<nentries<<std::endl;
  //size_t lumisecid=0;
  //unsigned int lumilumisecid=0;
  //runnumber=lumiheader->runNumber;
  //
  //hardcode the first LS is always alive
  //
  unsigned int ncollidingbunches=0;
  for(size_t i=0;i<nentries;++i){
    lumi::Lumi2DB::PerLumiData h;
    h.cmsalive=1;
    hlxtree->GetEntry(i);
    //std::cout<<"live flag "<<lumiheader->bCMSLive <<std::endl;
    if( !lumiheader->bCMSLive && i!=0){
      std::cout<<"\t non-CMS LS "<<lumiheader->sectionNumber<<" ";
      h.cmsalive=0;
    }
    ++ncmslumi;
    if(ncmslumi==1){//just take the first ls
      ncollidingbunches=lumiheader->numBunches;
    }
    h.bxET.reserve(lumi::N_BX);
    h.bxOCC1.reserve(lumi::N_BX);
    h.bxOCC2.reserve(lumi::N_BX);
    
    //runnumber=lumiheader->runNumber;
    //if(runnumber!=m_run) throw std::runtime_error(std::string("requested run ")+this->int2str(m_run)+" does not match runnumber in the data header "+this->int2str(runnumber));
    h.lumilsnr=lumiheader->sectionNumber;
    std::map<unsigned int , Lumi2DB::beamData >::iterator beamIt=dipmap.find(h.lumilsnr);
    if ( beamIt!=dipmap.end() ){
      h.beammode=beamIt->second.mode;
      h.beamenergy=beamIt->second.energy;
      h.nlivebx=beamIt->second.nlivebx;
      if(h.nlivebx!=0){
	h.bxindex=(short*)malloc(sizeof(short)*h.nlivebx);
	h.beamintensity_1=(float*)malloc(sizeof(float)*h.nlivebx);
	h.beamintensity_2=(float*)malloc(sizeof(float)*h.nlivebx);
	if(h.bxindex==0 || h.beamintensity_1==0 || h.beamintensity_2==0){
	  std::cout<<"malloc failed"<<std::endl;
	}
	//std::cout<<"h.bxindex size "<<sizeof(short)*h.nlivebx<<std::endl;
	//std::cout<<"h.beamintensity_1 size "<<sizeof(float)*h.nlivebx<<std::endl;
	//std::cout<<"h.beamintensity_2 size "<<sizeof(float)*h.nlivebx<<std::endl;

	std::memmove(h.bxindex,beamIt->second.bxindex,sizeof(short)*h.nlivebx);
	std::memmove(h.beamintensity_1,beamIt->second.beamintensity_1,sizeof(float)*h.nlivebx);
	std::memmove(h.beamintensity_2,beamIt->second.beamintensity_2,sizeof(float)*h.nlivebx);

	::free(beamIt->second.bxindex);beamIt->second.bxindex=0;
	::free(beamIt->second.beamintensity_1);beamIt->second.beamintensity_1=0;
	::free(beamIt->second.beamintensity_2);beamIt->second.beamintensity_2=0;
      }else{
	//std::cout<<"h.nlivebx is zero"<<std::endl;
	h.bxindex=0;
	h.beamintensity_1=0;
	h.beamintensity_2=0;
      }
    }else{
      h.beammode="N/A";
      h.beamenergy=0.0;
      h.nlivebx=0;
      h.bxindex=0;
      h.beamintensity_1=0;
      h.beamintensity_2=0;
    }
    h.startorbit=lumiheader->startOrbit;
    h.numorbit=lumiheader->numOrbits;
    if(h.cmsalive==0){
      h.cmslsnr=0; //the dead ls has cmsls number=0
    }else{
      h.cmslsnr=ncmslumi;//we guess cms lumils
    }
    h.instlumi=lumisummary->InstantLumi;
    //std::cout<<"instant lumi "<<lumisummary->InstantLumi<<std::endl;
    h.instlumierror=lumisummary->InstantLumiErr;
    h.lumisectionquality=lumisummary->InstantLumiQlty;
    h.dtnorm=lumisummary->DeadTimeNormalization;
    h.lhcnorm=lumisummary->LHCNormalization;
    //unsigned int timestp=lumiheader->timestamp;
    //std::cout<<"cmslsnum "<<ncmslumi<<"timestp "<<timestp<<std::endl;
    for(size_t i=0;i<lumi::N_BX;++i){
      lumi::Lumi2DB::PerBXData bET;
      lumi::Lumi2DB::PerBXData bOCC1;
      lumi::Lumi2DB::PerBXData bOCC2;
      //bET.idx=i+1;
      bET.lumivalue=lumidetail->ETLumi[i];
      bET.lumierr=lumidetail->ETLumiErr[i];
      bET.lumiquality=lumidetail->ETLumiQlty[i];      
      h.bxET.push_back(bET);

      //bOCC1.idx=i+1;

      bOCC1.lumivalue=lumidetail->OccLumi[0][i];
      bOCC1.lumierr=lumidetail->OccLumiErr[0][i];
      /**if(bOCC1.lumivalue*6.370>1.0e-04){
	if(i!=0){
	  std::cout<<i<<" detail "<<(i-1)*10+1<<" "<<(timestp-1262300400)<<" "<<bOCC1.lumivalue*6.37<<" "<<bOCC1.lumierr*6.37<<std::endl;
	}
      }
      **/
      bOCC1.lumiquality=lumidetail->OccLumiQlty[0][i]; 
      h.bxOCC1.push_back(bOCC1);
          
      //bOCC2.idx=i+1;
      bOCC2.lumivalue=lumidetail->OccLumi[1][i];
      bOCC2.lumierr=lumidetail->OccLumiErr[1][i];
      bOCC2.lumiquality=lumidetail->OccLumiQlty[1][i]; 
      h.bxOCC2.push_back(bOCC2);
    }
    lumiresult.push_back(h);
  }
  std::cout<<std::endl;
  if(nstablebeam!=0){
    bgev=bgev/nstablebeam;//nominal beam energy=sum(energy)/nstablebeams
  }
  std::cout<<"avg stable beam energy "<<bgev<<std::endl;
  if( !m_novalidate && !isLumiDataValid(lumiresult.begin(),lumiresult.end()) ){
    throw lumi::invalidDataException("all lumi values are <0.5e-08","isLumiDataValid","Lumi2DB");
  }
  if( !m_nocheckingstablebeam && !hasStableBeam(lumiresult.begin(),lumiresult.end()) ){
    throw lumi::noStableBeamException("no LS has STABLE BEAMS","hasStableBeam","Lumi2DB");
  }
  coral::ConnectionService* svc=new coral::ConnectionService;
  lumi::DBConfig dbconf(*svc);
  if(!m_authpath.empty()){
    dbconf.setAuthentication(m_authpath);
  }
  coral::ISessionProxy* session=svc->connect(m_dest,coral::Update);
  coral::ITypeConverter& tpc=session->typeConverter();
  tpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
  //
  //write to old lumisummary
  //
  try{
    const std::string lversion(filenamecontent.version);
    if(m_mode==std::string("beamintensity_only")){
      std::cout<<"writing beam intensity only to old lumisummary table "<<std::endl;
      writeBeamIntensityOnly(session,runnumber,lversion,lumiresult.begin(),lumiresult.end());
      std::cout<<"done"<<std::endl;
    }else{
       if(m_mode=="loadoldschema"){
	  std::cout<<"writing all lumi data to old lumisummary table "<<std::endl;
	  writeAllLumiData(session,runnumber,lversion,lumiresult.begin(),lumiresult.end());     
	  std::cout<<"done"<<std::endl;
       }
    }
    std::cout<<"writing all lumi data to lumisummary_V2 table "<<std::endl;
    lumidataid=writeAllLumiDataToSchema2(session,m_source,runnumber,bgev,ncollidingbunches,lumiresult.begin(),lumiresult.end());
    std::cout<<"done"<<std::endl;
    cleanTemporaryMemory(lumiresult.begin(),lumiresult.end());
    delete session;
    delete svc;
  }catch( const coral::Exception& er){
    std::cout<<"database error "<<er.what()<<std::endl;
    session->transaction().rollback();
    delete session;
    delete svc;
    throw er;
  }
  return lumidataid;
}
const std::string lumi::Lumi2DB::dataType() const{
  return "LUMI";
}
const std::string lumi::Lumi2DB::sourceType() const{
  return "DB";
}
lumi::Lumi2DB::~Lumi2DB(){}

#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::Lumi2DB,"Lumi2DB");
#endif

// -*- C++ -*-
//
// Package:    RPCNoiseStripsSH
// Class:      RPCNoiseStripsSH
// 
/**\class RPCNoiseStripsSH RPCNoiseStripsSH.cc CondTools/RPC/src/RPCNoiseStripsSH.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Simranjit Singh Chhibra,40 5-B06,+41227674539,
//         Created:  Fri Oct  7 10:51:47 CEST 2011
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/RPC/interface/RPCNoiseStripsSH.h"

#include "RelationalAccess/ISession.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"

#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"


popcon::RPCNoiseStripsSH::RPCNoiseStripsSH(const edm::ParameterSet& iConfig)
{
  
  m_host = iConfig.getUntrackedParameter<std::string>("host");
  m_user = iConfig.getUntrackedParameter<std::string>("user");
  m_passw = iConfig.getUntrackedParameter<std::string>("passw");
  m_version = iConfig.getUntrackedParameter<unsigned int>("version");
  m_run   = iConfig.getUntrackedParameter<unsigned int>("run");
  std::cout << "-------------START-------------" <<std::endl;
 
}


popcon::RPCNoiseStripsSH::~RPCNoiseStripsSH()
{
  std::cout << "-------------END-------------" <<std::endl;
}

void
popcon::RPCNoiseStripsSH::getNewObjects()
{
  
  
  std::cout<< "taginfo.name====="<<tagInfo().name<<
    "\t taginfo.size====="<<tagInfo().size<<
    "\t lastRun====="<<tagInfo().lastInterval.first<< std::endl;
  
  
  coral::ISession* masterSession = this->connect( m_host,m_user,m_passw );
  masterSession->transaction().start( true );
  
  coral::ISchema& masterSchema = masterSession->nominalSchema();
  coral::IQuery* masterQuery = masterSchema.newQuery();
  coral::IQueryDefinition& subQuery = masterQuery->defineSubQuery("SQ");
  
  masterQuery->addToTableList("SQ");
  masterQuery->addToOutputList("RUN_NUMBER");
  masterQuery->limitReturnedRows(m_run);
  subQuery.addToTableList("RPC_NOISE_STRIPS");
  subQuery.addToOutputList("RUN_NUMBER");
  subQuery.addToOutputList("RAW_ID");
  subQuery.addToOrderList("RUN_NUMBER");
  
  coral::AttributeList subCondition;
  subCondition.extend<int>("run");
  subCondition.extend<int>("rawid");
  
  subCondition["run"].data<int>() = tagInfo().lastInterval.first + 1;
  subCondition["rawid"].data<int>() = 637599878;
  std::string subcondition = "RUN_NUMBER >= :run and RAW_ID = :rawid";
  
  subQuery.setCondition(subcondition, subCondition);
  
  coral::ICursor& masterCursor = masterQuery->execute();
  
  std::vector<int> runs;
  while(masterCursor.next()){
    const coral::AttributeList& masterRow = masterCursor.currentRow();
    runs.push_back(masterRow["RUN_NUMBER"].data<long long int>());
    //std::cout<< "masterRow[RUN_NUMBER]" <<masterRow["RUN_NUMBER"].data<long long int>()<<std::endl;
  }
  
  for (std::vector<int>::iterator master_run=runs.begin();master_run<runs.end();master_run++){
    
    int n_run = * master_run;
    
    coral::ISession* session = this->connect( m_host,m_user,m_passw );
    session->transaction().start( true );
    
    coral::ISchema& schema = session->nominalSchema();
    
    coral::IQuery* query = schema.newQuery();
    
    query->addToTableList("RPC_NOISE_STRIPS");
    query->addToOutputList("RUN_NUMBER");
    query->addToOutputList("RAW_ID");
    query->addToOutputList("CHANNEL_NUMBER");
    query->addToOutputList("STRIP_NUMBER");
    query->addToOutputList("IS_DEAD");
    query->addToOutputList("IS_MASKED");
    query->addToOutputList("RATE_HZ_CM2");
    
    coral::AttributeList conditionData;
    conditionData.extend<int>("run");
    
    conditionData["run"].data<int>() = n_run;
    std::string condition = "RUN_NUMBER = :run";
    
    query->setCondition(condition, conditionData);
    
    coral::ICursor& cursor = query->execute();
    
    rpcNoiseStrips = new RPCNoiseStripsObject();
    rpcNoiseStrips->run = n_run;
    rpcNoiseStrips->version = m_version;
    
    std::vector<RPCNoiseStripsObject::NoiseStripsObjectItem> ItemVector;
    
    
    while(cursor.next()){
      RPCNoiseStripsObject::NoiseStripsObjectItem NSOI;
      
      
      const coral::AttributeList& row = cursor.currentRow();
      NSOI.dpid = row["RAW_ID"].data<long long int>();
      NSOI.channelNumber = row["CHANNEL_NUMBER"].data<long long int>();
      NSOI.stripNumber = row["STRIP_NUMBER"].data<long long int>();
      NSOI.isDead = row["IS_DEAD"].data<long long int>();
      NSOI.isMasked = row["IS_MASKED"].data<long long int>();
      NSOI.rate = row["RATE_HZ_CM2"].data<double>();
      NSOI.weight = 1;
      
      //std::cout<< "dpid of NSOI======="<<NSOI.dpid<<std::endl;
      
      ItemVector.push_back(NSOI);
    }
    
    std::cout <<"Run = "<<n_run<<" number of Items "<<ItemVector.size()<<std::endl;
    rpcNoiseStrips->v_cls = ItemVector;
    delete query;
    
    session->transaction().commit();
    delete session;
    
    if (ItemVector.size()!=0){
      m_to_transfer.push_back(std::make_pair(rpcNoiseStrips,n_run)); 
    }
  }
}




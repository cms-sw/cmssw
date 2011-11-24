// -*- C++ -*-
//
// Package:    DoAQuerySH
// Class:      DoAQuerySH
// 
/**\class DoAQuerySH DoAQuerySH.cc CondTools/DoAQuery/src/DoAQuerySH.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Simranjit Singh Chhibra,40 5-B06,+41227674539,
//         Created:  Fri Oct  7 10:51:47 CEST 2011
// $Id: DoAQuerySH.cc,v 1.1 2011/11/07 08:20:47 mmaggi Exp $
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
#include "CondTools/RPC/interface/DoAQuerySH.h"

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
#include "CoralBase/TimeStamp.h"

popcon::DoAQuerySH::DoAQuerySH(const edm::ParameterSet& iConfig)
{
  m_host = iConfig.getUntrackedParameter<std::string>("host");
  m_user = iConfig.getUntrackedParameter<std::string>("user");
  m_passw = iConfig.getUntrackedParameter<std::string>("passw");
  std::cout << "-------------START-------------" <<std::endl;
}


popcon::DoAQuerySH::~DoAQuerySH()
{
  std::cout << "-------------END-------------" <<std::endl;
}

void
popcon::DoAQuerySH::getNewObjects()
{

  std::cout<< "taginfo.name====="<<tagInfo().name<<
    "\t taginfo.size====="<<tagInfo().size<<
    "\t lastRun====="<<tagInfo().lastInterval.first<< std::endl;
  
  coral::ISession* masterSession = this->connect( m_host,m_user,m_passw );
  masterSession->transaction().start( true );

  coral::ISchema& masterSchema = masterSession->nominalSchema();
  coral::IQuery* masterQuery = masterSchema.newQuery();
  coral::IQueryDefinition& subQuery = masterQuery->defineSubQuery("ALIAS");


  masterQuery->addToTableList("ALIAS","A");
  masterQuery->addToTableList("RPCPVSSDETID","E");
  masterQuery->addToOutputList("A.L");
  masterQuery->addToOutputList("A.T");
  masterQuery->addToOutputList("E.PVSS_ID");
  masterQuery->addToOutputList("E.REGION");
  masterQuery->addToOutputList("E.RING");
  masterQuery->addToOutputList("E.STATION");
  masterQuery->addToOutputList("E.SECTOR");
  masterQuery->addToOutputList("E.LAYER");
  masterQuery->addToOutputList("E.SUBSECTOR");
  masterQuery->addToOutputList("E.SUPPLYTYPE");
  masterQuery->addToOrderList("A.L");

  subQuery.addToTableList("RPCPVSSDETID","E");
  subQuery.addToTableList("DP_NAME2ID","B");
  subQuery.addToTableList("ALIASES","A");
  subQuery.addToOutputList("A.ALIAS","L");
  subQuery.addToOutputList("max(E.SINCE)","T");
  subQuery.groupBy("A.ALIAS");

  coral::AttributeList subCondition;
  subCondition.extend<coral::TimeStamp>("tsince");
  subCondition.extend<int>("region");
  subCondition.extend<int>("ring");
  subCondition.extend<std::string>("suptype");
  
  coral::TimeStamp tsince(2012,1,9,10,0,0,0);
  subCondition["tsince"].data<coral::TimeStamp>() = tsince;
  subCondition["suptype"].data<std::string>() = "T";
  subCondition["region"].data<int>() = 9;
  subCondition["ring"].data<int>() = 9;
  std::string subcondition1 = "E.PVSS_ID=B.ID and A.DPE_NAME=B.DPNAME";
  std::string subcondition2 = " and E.SUPPLYTYPE=:suptype";
  std::string subcondition3 = " and E.SINCE < :tsince";
  std::string subcondition4 = " and E.REGION<:region and E.RING<:ring";
  std::string subcondition  = subcondition1+
    subcondition2+
    subcondition3+
    subcondition4;
    
  subQuery.setCondition(subcondition, subCondition);

  
  coral::AttributeList masCondition;
  masterQuery->setCondition("A.T=E.SINCE",masCondition);

  coral::ICursor& cur = masterQuery->execute();


  int nrow=0;
  while(cur.next()){
    nrow++;
    const coral::AttributeList& row = cur.currentRow();
    std::string alias(row["A.L"].data<std::string>());
    coral::TimeStamp ts = row["A.T"].data<coral::TimeStamp>();
    int pdid=row["E.PVSS_ID"].data<int>();
    std::string regi=row["E.REGION"].data<std::string>();
    std::string ring=row["E.RING"].data<std::string>();
    std::string stat=row["E.STATION"].data<std::string>();
    std::string sect=row["E.SECTOR"].data<std::string>();
    std::string laye=row["E.LAYER"].data<std::string>();
    std::string subs=row["E.SUBSECTOR"].data<std::string>();
    std::string typ(row["E.SUPPLYTYPE"].data<std::string>());
    int reg=atoi(regi.c_str());
    int rin=atoi(ring.c_str());
    int sta=atoi(stat.c_str());
    int sec=atoi(sect.c_str());
    int lay=atoi(laye.c_str());
    int sub=atoi(subs.c_str());
    std::cout<<"-> raw number="<<nrow
	     <<" alias="<<alias
	     <<" date="<<ts.day()<<"/"<<ts.month()<<"/"<<ts.year()<<" "
	     <<ts.hour()<<":"<<ts.minute()<<"."<<ts.second()
	     <<" pdis="<<pdid
	     <<" region="<<reg
	     <<" ring="<<rin
	     <<" station="<<sta
	     <<" sector="<<sec
	     <<" layer="<<lay
	     <<" subsector="<<sub
	     <<std::endl;
  }

  delete masterQuery;
    
  masterSession->transaction().commit();
  delete masterSession;
  
}




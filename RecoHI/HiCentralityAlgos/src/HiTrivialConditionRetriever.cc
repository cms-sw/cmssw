// -*- C++ -*-
//
// Package:    HiTrivialConditionRetriever
// Class:      HiTrivialConditionRetriever
// 
/**\class HiTrivialConditionRetriever HiTrivialConditionRetriever.cc yetkin/HiTrivialConditionRetriever/src/HiTrivialConditionRetriever.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Wed May  2 21:41:30 EDT 2007
// $Id: HiTrivialConditionRetriever.cc,v 1.1 2010/03/23 21:56:39 yilmaz Exp $
//
//


// system include files
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"

using namespace std;

//
// class decleration
//

class HiTrivialConditionRetriever : public edm::ESProducer, 
				public edm::EventSetupRecordIntervalFinder
{
public:
  HiTrivialConditionRetriever(const edm::ParameterSet&);

protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue& ,
                               edm::ValidityInterval& ) ;
  
   private:
  virtual std::auto_ptr<CentralityTable> produceTable( const HeavyIonRcd& );
  void printBin(const CentralityTable::CBin*);

  // ----------member data ---------------------------

 int verbose_;
   string inputFileName_;   
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HiTrivialConditionRetriever::HiTrivialConditionRetriever(const edm::ParameterSet& iConfig){
  
  setWhatProduced(this, &HiTrivialConditionRetriever::produceTable);
  findingRecord<HeavyIonRcd>();
  
  //now do what ever initialization is needed
  verbose_ = iConfig.getUntrackedParameter<int>("verbosity",1);
  inputFileName_ = iConfig.getParameter<string>("inputFile");
}

std::auto_ptr<CentralityTable> 
HiTrivialConditionRetriever::produceTable( const HeavyIonRcd& ){

  std::auto_ptr<CentralityTable> CT(new CentralityTable()) ;

  // Get values from text file
  ifstream in( edm::FileInPath(inputFileName_).fullPath().c_str() );
  string line;

  int i = 0;
  while ( getline( in, line ) ) {
    if ( !line.size() || line[0]=='#' ) { continue; }
    CentralityTable::CBin thisBin;
    CT->m_table.push_back(thisBin);
    istringstream ss(line);
    ss>>CT->m_table[i].bin_edge
      >>CT->m_table[i].n_part.mean
      >>CT->m_table[i].n_part.var
      >>CT->m_table[i].n_coll.mean
      >>CT->m_table[i].n_coll.var
      >>CT->m_table[i].n_hard.mean
      >>CT->m_table[i].n_hard.var
      >>CT->m_table[i].b.mean
      >>CT->m_table[i].b.var;
    i++;
  }

  return CT; 
}

void HiTrivialConditionRetriever::printBin(const CentralityTable::CBin* thisBin){
   cout<<"HF Cut = "<<thisBin->bin_edge<<endl;
   cout<<"Npart = "<<thisBin->n_part.mean<<endl;
   cout<<"sigma = "<<thisBin->n_part.var<<endl;
   cout<<"Ncoll = "<<thisBin->n_coll.mean<<endl;
   cout<<"sigma = "<<thisBin->n_coll.var<<endl;
   cout<<"B     = "<<thisBin->b.mean<<endl;
   cout<<"sigma = "<<thisBin->b.var<<endl;
   cout<<"__________________________________________________"<<endl;
}

void
HiTrivialConditionRetriever::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& rk,
					     const edm::IOVSyncValue& iTime,
					     edm::ValidityInterval& oValidity)
{
  if(verbose_>=1) std::cout << "HiTrivialConditionRetriever::setIntervalFor(): record key = " << rk.name() << "\ttime: " << iTime.time().value() << std::endl;
  //For right now, we will just use an infinite interval of validity
  oValidity = edm::ValidityInterval( edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime() );
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(HiTrivialConditionRetriever);

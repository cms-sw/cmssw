// -*- C++ -*-
//
// Package:    BTagPerformaceRootProducerFromSQLITE
// Class:      BTagPerformaceRootProducerFromSQLITE
// 
/**\class BTagPerformaceRootProducerFromSQLITE BTagPerformaceRootProducerFromSQLITE.cc junk/BTagPerformaceRootProducerFromSQLITE/src/BTagPerformaceRootProducerFromSQLITE.cc

 Description: This writes out a ROOT file with a BtagPerformance object taken from an sqlite file. 

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Thu Feb 11 14:21:59 CST 2010
// $Id: BTagPerformaceRootProducerFromSQLITE.cc,v 1.3 2010/06/04 18:04:31 srappocc Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "PhysicsTools/CondLiteIO/interface/RecordWriter.h"
#include "DataFormats/Provenance/interface/ESRecordAuxiliary.h"
#include "DataFormats/FWLite/interface/format_type_name.h"

#include "RecoBTag/Records/interface/BTagPerformanceRecord.h"
#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"

#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"



//
// class declaration
//

class BTagPerformaceRootProducerFromSQLITE : public edm::EDAnalyzer {
   public:
      explicit BTagPerformaceRootProducerFromSQLITE(const edm::ParameterSet&);
      ~BTagPerformaceRootProducerFromSQLITE();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
  std::vector<std::string>  names_;
  edm::ESWatcher<BTagPerformanceRecord> recWatcher_;
  std::auto_ptr<fwlite::RecordWriter> writer_;
  edm::IOVSyncValue lastValue_;
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
BTagPerformaceRootProducerFromSQLITE::BTagPerformaceRootProducerFromSQLITE(const edm::ParameterSet& iConfig) :
  names_(iConfig.getParameter< std::vector<std::string> >("names"))
{
}


BTagPerformaceRootProducerFromSQLITE::~BTagPerformaceRootProducerFromSQLITE()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
BTagPerformaceRootProducerFromSQLITE::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::cout << "Hello from BTagPerformaceRootProducerFromSQLITE!" << std::endl;
  if(recWatcher_.check(iSetup)) {
    const BTagPerformanceRecord& r = iSetup.get<BTagPerformanceRecord>();

    if(! writer_.get()) {
      edm::Service<TFileService> fs ;
      TFile * f = &(fs->file());
      writer_ = std::auto_ptr<fwlite::RecordWriter>(new fwlite::RecordWriter(r.key().name(), f ));
    }
    lastValue_ = r.validityInterval().last();

    for( size_t i=0;  i<names_.size(); i++ )
    {
       edm::ESHandle<BtagPerformance> perfH;
       std::cout <<" Studying performance with label " << names_.at(i) <<std::endl;
       r.get( names_.at(i), perfH );
       const BtagPerformance & perf = *(perfH.product());

       writer_->update(&(perf.payload()), typeid(PerformancePayload),names_.at(i).c_str());
       writer_->update(&(perf.workingPoint()), typeid(PerformanceWorkingPoint),names_.at(i).c_str());
    }
    writer_->fill(edm::ESRecordAuxiliary(r.validityInterval().first().eventID(),
                                      r.validityInterval().first().time()));
  }
}


// ------------ method called once each job just before starting event loop  ------------
void 
BTagPerformaceRootProducerFromSQLITE::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BTagPerformaceRootProducerFromSQLITE::endJob() {
  /*if(writer_.get()) {
    writer_->fill(edm::ESRecordAuxiliary(lastValue_.eventID(),
                                         lastValue_.time()));
    writer_->write();
  } */
}

//define this as a plug-in
DEFINE_FWK_MODULE(BTagPerformaceRootProducerFromSQLITE);

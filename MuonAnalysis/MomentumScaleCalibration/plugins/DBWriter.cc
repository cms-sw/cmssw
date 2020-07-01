// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DBWriter.h"
#include "CondFormats/RecoMuonObjects/interface/MuScleFitDBobject.h"

DBWriter::DBWriter(const edm::ParameterSet& ps) {
  // This string is one of: scale, resolution, background.
  std::string type(ps.getUntrackedParameter<std::string>("Type"));
  // Create the corrector and set the parameters
  if (type == "scale")
    corrector_ = std::make_unique<MomentumScaleCorrector>(ps.getUntrackedParameter<std::string>("CorrectionsIdentifier"));
  else if (type == "resolution")
    corrector_ = std::make_unique<ResolutionFunction>(ps.getUntrackedParameter<std::string>("CorrectionsIdentifier"));
  else if (type == "background")
    corrector_ = std::make_unique<BackgroundFunction>(ps.getUntrackedParameter<std::string>("CorrectionsIdentifier"));
  else {
    std::cout << "Error: unrecognized type. Use one of those: 'scale', 'resolution', 'background'" << std::endl;
    exit(1);
  }
}

DBWriter::~DBWriter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to for each event  ------------
void DBWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  MuScleFitDBobject* dbObject = new MuScleFitDBobject;

  dbObject->identifiers = corrector_->identifiers();
  dbObject->parameters = corrector_->parameters();

  //   if( dbObject->identifiers.size() != dbObject->parameters.size() ) {
  //     std::cout << "Error: size of parameters("<<dbObject->parameters.size()<<") and identifiers("<<dbObject->identifiers.size()<<") don't match" << std::endl;
  //     exit(1);
  //   }

  //   std::vector<std::vector<double> >::const_iterator parVec = dbObject->parameters.begin();
  //   std::vector<int>::const_iterator id = dbObject->identifiers.begin();
  //   for( ; id != dbObject->identifiers.end(); ++id, ++parVec ) {
  //     std::cout << "id = " << *id << std::endl;
  //     std::vector<double>::const_iterator par = parVec->begin();
  //     int i=0;
  //     for( ; par != parVec->end(); ++par, ++i ) {
  //       std::cout << "par["<<i<<"] = " << *par << std::endl;
  //     }
  //   }

  // Save the parameters to the db.
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("MuScleFitDBobjectRcd")) {
      mydbservice->createNewIOV<MuScleFitDBobject>(
          dbObject, mydbservice->beginOfTime(), mydbservice->endOfTime(), "MuScleFitDBobjectRcd");
    } else {
      mydbservice->appendSinceTime<MuScleFitDBobject>(dbObject, mydbservice->currentTime(), "MuScleFitDBobjectRcd");
    }
  } else {
    edm::LogError("DBWriter") << "Service is unavailable" << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DBWriter);

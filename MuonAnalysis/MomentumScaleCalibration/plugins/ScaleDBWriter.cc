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

#include "ScaleDBWriter.h"
// #include "MuonAnalysis/MomentumScaleCalibrationObjects/interface/MuScleFitScale.h"
#include "CondFormats/MomentumScaleCalibrationObjects/interface/MuScleFitScale.h"

using namespace std;
using namespace edm;

ScaleDBWriter::ScaleDBWriter(const edm::ParameterSet& ps)

{
  // Create the corrector and set the parameters
  corrector_.reset(new MomentumScaleCorrector( ps.getUntrackedParameter<string>("CorrectionsIdentifier") ) );
}

ScaleDBWriter::~ScaleDBWriter()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

// ------------ method called to for each event  ------------
void
ScaleDBWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  MuScleFitScale * scaleObject = new MuScleFitScale;

  scaleObject->identifiers = corrector_->identifiers();
  scaleObject->parameters = corrector_->parameters();

  if( scaleObject->identifiers.size() != scaleObject->parameters.size() ) {
    cout << "Error: size of parameters("<<scaleObject->parameters.size()<<") and identifiers("<<scaleObject->identifiers.size()<<") don't match" << endl;
    exit(1);
  }
//   vector<vector<double> >::const_iterator parVec = scaleObject->parameters.begin();
//   vector<int>::const_iterator id = scaleObject->identifiers.begin();
//   for( ; id != scaleObject->identifiers.end(); ++id, ++parVec ) {
//     cout << "id = " << *id << endl;
//     vector<double>::const_iterator par = parVec->begin();
//     int i=0;
//     for( ; par != parVec->end(); ++par, ++i ) {
//       cout << "par["<<i<<"] = " << *par << endl;
//     }
//   }

  // Save the parameters to the db.
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( mydbservice.isAvailable() ){
    if( mydbservice->isNewTagRequest("MuScleFitScaleRcd") ){
      mydbservice->createNewIOV<MuScleFitScale>(scaleObject,mydbservice->beginOfTime(),mydbservice->endOfTime(),"MuScleFitScaleRcd");
    } else {
      mydbservice->appendSinceTime<MuScleFitScale>(scaleObject,mydbservice->currentTime(),"MuScleFitScaleRcd");      
    }
  } else {
    edm::LogError("ScaleDBWriter")<<"Service is unavailable"<<std::endl;
  }

}

// ------------ method called once each job just before starting event loop  ------------
void 
ScaleDBWriter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ScaleDBWriter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(ScaleDBWriter);

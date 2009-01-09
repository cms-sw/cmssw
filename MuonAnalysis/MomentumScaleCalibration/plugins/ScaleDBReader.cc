// #include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
// #include "MuonAnalysis/MomentumScaleCalibrationObjects/interface/MuScleFitScale.h"
#include "CondFormats/MomentumScaleCalibrationObjects/interface/MuScleFitScale.h"
// #include "MuonAnalysis/MomentumScaleCalibrationObjects/interface/MuScleFitScaleRcd.h"
#include "CondFormats/DataRecord/interface/MuScleFitScaleRcd.h"

#include "ScaleDBReader.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <string>

using namespace std;
using namespace cms;

ScaleDBReader::ScaleDBReader( const edm::ParameterSet& iConfig ){}

void ScaleDBReader::beginJob ( const edm::EventSetup& iSetup ) {
  edm::ESHandle<MuScleFitScale> scaleObject;
  iSetup.get<MuScleFitScaleRcd>().get(scaleObject);
  edm::LogInfo("ScaleDBReader") << "[ScaleDBReader::analyze] End Reading MuScleFitScaleRcd" << endl;

  cout << "identifiers size from scaleObject = " << scaleObject->identifiers.size() << endl;
  cout << "parameters size from scaleObject = " << scaleObject->parameters.size() << endl;;

  // Create the corrector and set the parameters
  corrector_.reset(new MomentumScaleCorrector( scaleObject.product() ) );

  cout << "pointer = " << corrector_.get() << endl;
}

//:  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

ScaleDBReader::~ScaleDBReader(){}

void ScaleDBReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  cout << "checking size consistency" << endl;
  if( corrector_->identifiers().size() != corrector_->parameters().size() ) {
    cout << "Error: size of parameters("<<corrector_->parameters().size()<<") and identifiers("<<corrector_->identifiers().size()<<") don't match" << endl;
    exit(1);
  }

  // Looping directly on it does not work, because it is returned by value
  // and the iterator gets invalidated on the next line. Save it to a temporary object
  // and iterate on it.
  vector<vector<double> > parVecVec(corrector_->parameters());
  // vector<vector<double> >::const_iterator parVec = corrector_->parameters().begin();
  vector<vector<double> >::const_iterator parVec = parVecVec.begin();
  vector<int>::const_iterator id = corrector_->identifiers().begin();
  for( ; id != corrector_->identifiers().end(); ++id, ++parVec ) {
    cout << "parVec.size() = " << parVec->size() << endl;
    cout << "parVec[0] = " << (*parVec)[0] << endl;
    cout << "id = " << *id << endl;
    vector<double>::const_iterator par = parVec->begin();
    int i=0;
    for( ; par != parVec->end(); ++par, ++i ) {
      cout << "par["<<i<<"] = " << *par << endl;
    }
  }

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(ScaleDBReader);

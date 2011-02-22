#ifndef DBReader_H
#define DBReader_H

// system include files
//#include <memory>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundFunction.h"

using namespace std;

class DBReader : public edm::EDAnalyzer
{
 public:
  explicit DBReader( const edm::ParameterSet& );
  ~DBReader();

  void initialize( const edm::EventSetup& iSetup );

  void analyze( const edm::Event&, const edm::EventSetup& );

 private:

  template <typename T>
  void printParameters(const T & functionPtr)
  {
    // Looping directly on it does not work, because it is returned by value
    // and the iterator gets invalidated on the next line. Save it to a temporary object
    // and iterate on it.
    vector<double> parVecVec(functionPtr->parameters());
    vector<double>::const_iterator parVec = parVecVec.begin();
    vector<int> functionId(functionPtr->identifiers());
    vector<int>::const_iterator id = functionId.begin();
    cout << "total number of parameters read from database = parVecVec.size() = " << parVecVec.size() << endl;
    int iFunc = 0;
    for( ; id != functionId.end(); ++id, ++iFunc ) {
      int parNum = functionPtr->function(iFunc)->parNum();
      cout << "For function id = " << *id << ", with "<<parNum<< " parameters: " << endl;
      for( int par=0; par<parNum; ++par ) {
        cout << "par["<<par<<"] = " << *parVec << endl;
        ++parVec;
      }
    }
  }

  //  uint32_t printdebug_;
  string type_;
  //auto_ptr<BaseFunction> corrector_;
  boost::shared_ptr<MomentumScaleCorrector> corrector_;
  boost::shared_ptr<ResolutionFunction> resolution_;
  boost::shared_ptr<BackgroundFunction> background_;
};
#endif

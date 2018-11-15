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

class DBReader : public edm::EDAnalyzer
{
 public:
  explicit DBReader( const edm::ParameterSet& );
  ~DBReader() override;

  void initialize( const edm::EventSetup& iSetup );

  void analyze( const edm::Event&, const edm::EventSetup& ) override;

 private:

  template <typename T>
  void printParameters(const T & functionPtr)
  {
    // Looping directly on it does not work, because it is returned by value
    // and the iterator gets invalidated on the next line. Save it to a temporary object
    // and iterate on it.
    std::vector<double> parVecVec(functionPtr->parameters());
    std::vector<double>::const_iterator parVec = parVecVec.begin();
    std::vector<int> functionId(functionPtr->identifiers());
    std::vector<int>::const_iterator id = functionId.begin();
    std::cout << "total number of parameters read from database = parVecVec.size() = " << parVecVec.size() << std::endl;
    int iFunc = 0;
    for( ; id != functionId.end(); ++id, ++iFunc ) {
      int parNum = functionPtr->function(iFunc)->parNum();
      std::cout << "For function id = " << *id << ", with "<<parNum<< " parameters: " << std::endl;
      for( int par=0; par<parNum; ++par ) {
	std::cout << "par["<<par<<"] = " << *parVec << std::endl;
        ++parVec;
      }
    }
  }

  //  uint32_t printdebug_;
  std::string type_;
  //std::unique_ptr<BaseFunction> corrector_;
  boost::shared_ptr<MomentumScaleCorrector> corrector_;
  boost::shared_ptr<ResolutionFunction> resolution_;
  boost::shared_ptr<BackgroundFunction> background_;
};
#endif

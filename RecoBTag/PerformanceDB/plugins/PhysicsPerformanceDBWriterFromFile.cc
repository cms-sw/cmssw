#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/BTagPerformancePayloadRecord.h"
#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTableEtaJetEt.h"
#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTableEtaJetEtPhi.h"
#include "CondFormats/BTauObjects/interface/BtagCorrectionPerformancePayloadFromTableEtaJetEt.h"

class PhysicsPerformanceDBWriterFromFile : public edm::EDAnalyzer
{
public:
  PhysicsPerformanceDBWriterFromFile(const edm::ParameterSet&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}
  virtual void endJob() {}
  ~PhysicsPerformanceDBWriterFromFile() {}

private:
  std::string inputTxtFile;
};

PhysicsPerformanceDBWriterFromFile::PhysicsPerformanceDBWriterFromFile
  (const edm::ParameterSet& p)
{
  inputTxtFile = p.getUntrackedParameter<std::string>("inputTxtFile");
}

void PhysicsPerformanceDBWriterFromFile::beginJob(const edm::EventSetup&)
{
  //
  // read object from file
  //

  //
  // File Format is
  // - concrete class name
  // - description
  // - stride
  // - vector<float>
  //

  std::ifstream in;
  in.open(inputTxtFile.c_str());
  std::string concreteType;
  std::string comment;
  std::vector<float> pl;
  int stride;

  in >> concreteType;
  std::cout << "concrete Type is "<<concreteType<<std::endl;

  in>>stride;
  

  std::cout << "Stride is "<<stride<<std::endl;
  
  in>> comment;

  std::cout << "Comment is "<<comment<<std::endl;
  //  return ;

  int number=0;
  while (!in.eof()){
    float temp;
    in >> temp;
    std::cout <<" Intersing "<<temp<< " in position "<<number<<std::endl;
    number++;
    pl.push_back(temp);
  }

  //
  // CHECKS
  //
  if ((number % stride) != 0){
    std::cout <<" Table not well formed"<<std::endl;
  }

  in.close();

  //
  // now create pl etc etc
  //

  BtagPerformancePayload * btagpl = 0;

  if (concreteType == "BtagCorrectionPerformancePayloadFromTableEtaJetEt"){
    btagpl = new BtagCorrectionPerformancePayloadFromTableEtaJetEt(stride, comment, pl);
  }else if (concreteType == "BtagPerformancePayloadFromTableEtaJetEt"){
    btagpl = new BtagPerformancePayloadFromTableEtaJetEt(stride, comment, pl);
  }else if (concreteType == "BtagPerformancePayloadFromTableEtaJetEtPhi") {
    btagpl = new BtagPerformancePayloadFromTableEtaJetEtPhi(stride, comment, pl);
  }else{
    std::cout <<" Non existing request: " <<concreteType<<std::endl;
  }
  
  std::cout <<" Created the "<<concreteType <<" object"<<std::endl;
  
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable())
    {
      if (s->isNewTagRequest("BTagPerformancePayloadRecord"))
	{
	  s->createNewIOV<BtagPerformancePayload>(btagpl,
						  s->beginOfTime(),
						  s->endOfTime(),
						  "BTagPerformancePayloadRecord");
	}
      else
	{
	  
	  s->appendSinceTime<BtagPerformancePayload>(btagpl,
						     s->currentTime(),
						     "BTagPerformancePayloadRecord");
	}
    }
}

DEFINE_FWK_MODULE(PhysicsPerformanceDBWriterFromFile);

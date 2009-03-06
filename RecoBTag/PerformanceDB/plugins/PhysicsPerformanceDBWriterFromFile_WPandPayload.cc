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
#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTableEtaJetEtOnlyBeff.h"
#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTableEtaJetEtPhi.h"
#include "CondFormats/BTauObjects/interface/BtagCorrectionPerformancePayloadFromTableEtaJetEt.h"

#include "CondFormats/DataRecord/interface/BTagPerformanceWPRecord.h"
#include "CondFormats/BTauObjects/interface/BtagWorkingPoint.h"


class PhysicsPerformanceDBWriterFromFile_WPandPayload : public edm::EDAnalyzer
{
public:
  PhysicsPerformanceDBWriterFromFile_WPandPayload(const edm::ParameterSet&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}
  virtual void endJob() {}
  ~PhysicsPerformanceDBWriterFromFile_WPandPayload() {}

private:
  std::string inputTxtFile;
  std::string rec1,rec2;
};

PhysicsPerformanceDBWriterFromFile_WPandPayload::PhysicsPerformanceDBWriterFromFile_WPandPayload
  (const edm::ParameterSet& p)
{
  inputTxtFile = p.getUntrackedParameter<std::string>("inputTxtFile");
  rec1 = p.getUntrackedParameter<std::string>("RecordPayload");
  rec2 = p.getUntrackedParameter<std::string>("RecordWP");
}

void PhysicsPerformanceDBWriterFromFile_WPandPayload::beginJob(const edm::EventSetup&)
{
  //
  // read object from file
  //

  //
  // File Format is
  // - tagger name
  // - cut
  // - concrete class name
  // - description
  // - stride
  // - vector<float>
  //

  std::ifstream in;
  in.open(inputTxtFile.c_str());
  std::string tagger;
  float cut;
 
  std::string concreteType;
  std::string comment;
  std::vector<float> pl;
  int stride;

  in >> tagger;
  std::cout << "WP Tagger is "<<tagger<<std::endl;
  
  in >> cut;
  std::cout << "WP Cut is "<<cut<<std::endl;

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


  /*  for (int k=0;k<(number/stride); k++){
    for (int j=0; j<stride; j++){
      std::cout << "Pos["<<k<<","<<j<<"] = "<<pl[k*stride+j]<<std::endl;
    }
  }
  */

  //
  // now create pl etc etc
  //

  BtagWorkingPoint * wp = new BtagWorkingPoint(cut, tagger);

  BtagPerformancePayload * btagpl = 0;

  if (concreteType == "BtagCorrectionPerformancePayloadFromTableEtaJetEt"){
    btagpl = new BtagCorrectionPerformancePayloadFromTableEtaJetEt(stride, comment, pl);
  }else if (concreteType == "BtagPerformancePayloadFromTableEtaJetEt"){
    btagpl = new BtagPerformancePayloadFromTableEtaJetEt(stride, comment, pl);
  }else if (concreteType == "BtagPerformancePayloadFromTableEtaJetEtPhi") {
    btagpl = new BtagPerformancePayloadFromTableEtaJetEtPhi(stride, comment, pl);
  }else if (concreteType == "BtagPerformancePayloadFromTableEtaJetEtOnlyBeff") {
    btagpl = new BtagPerformancePayloadFromTableEtaJetEtOnlyBeff(stride, comment, pl);
  }else{
    std::cout <<" Non existing request: " <<concreteType<<std::endl;
  }
  
  std::cout <<" Created the "<<concreteType <<" object"<<std::endl;
  
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable())
    {
      if (s->isNewTagRequest(rec1))
	{
	  s->createNewIOV<BtagPerformancePayload>(btagpl,
						  s->beginOfTime(),
						  s->endOfTime(),
						  rec1);
	}
      else
	{
	  
	  s->appendSinceTime<BtagPerformancePayload>(btagpl,
						     // JUST A STUPID PATCH
						     111,
						     rec1);
	}
    }

  // write also the WP
  
  if (s.isAvailable())
    {
      if (s->isNewTagRequest(rec2))
	{
	  s->createNewIOV<BtagWorkingPoint>(wp,
					    s->beginOfTime(),
					    s->endOfTime(),
					    rec2);
	}
      else
	{
	  
	  s->appendSinceTime<BtagWorkingPoint>(wp,
					       /// JUST A STUPID PATCH
					       111,
					       rec2);
	}
    }
  
  
  
}

DEFINE_FWK_MODULE(PhysicsPerformanceDBWriterFromFile_WPandPayload);

// -*- C++ -*-

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
#include "CondFormats/JetMETObjects/interface/METCorrectorParameters.h"

//____________________________________________________________________________||
class  METCorrectorDBWriter : public edm::EDAnalyzer
{
 public:
  METCorrectorDBWriter(const edm::ParameterSet&);
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override {}
  virtual void endJob() override {}
  ~METCorrectorDBWriter() {}

 private:
  std::string era;
  std::string algo;
  std::string inputTxtFile;
  std::string payloadTag;
};

METCorrectorDBWriter::METCorrectorDBWriter(const edm::ParameterSet& pSet)
{
  era    = pSet.getUntrackedParameter<std::string>("era");
  algo   = pSet.getUntrackedParameter<std::string>("algo");
  payloadTag = algo;
}

void METCorrectorDBWriter::beginJob()
{
  std::string path("CondFormats/JetMETObjects/data/");

  std::cout << "Starting to import payload " << payloadTag << " from text files." << std::endl;

  std::string append("_");
  append += algo;
  append += ".txt";
  inputTxtFile = path + era + append;
  std::cout << " inputTxtFile " << inputTxtFile << std::endl;
  std::ifstream input( ("../../../"+inputTxtFile).c_str() );
  edm::FileInPath fip(inputTxtFile);
  std::string mSection = "";
  METCorrectorParameters *payload = new METCorrectorParameters(fip.fullPath(),mSection);
  payload->printScreen();
  if ( input.good() ) {
    edm::FileInPath fip(inputTxtFile);
    std::cout << "Opened file " << inputTxtFile << std::endl;
  }
  std::cout << "Opening PoolDBOutputService" << std::endl;

  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) 
    {
      std::cout << "Setting up payload tag " << payloadTag << std::endl;
      if (s->isNewTagRequest(payloadTag)) 
        s->createNewIOV<METCorrectorParameters>(payload, s->beginOfTime(), s->endOfTime(), payloadTag);
      else 
        s->appendSinceTime<METCorrectorParameters>(payload, 111, payloadTag);
    }
  std::cout << "Wrote in CondDB payload label: " << payloadTag << std::endl;
  
}

//____________________________________________________________________________||
DEFINE_FWK_MODULE(METCorrectorDBWriter);

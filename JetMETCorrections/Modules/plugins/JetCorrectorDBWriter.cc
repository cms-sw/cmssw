// Author: Benedikt Hegner
// Email:  benedikt.hegner@cern.ch

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
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

class  JetCorrectorDBWriter : public edm::EDAnalyzer
{
 public:
  JetCorrectorDBWriter(const edm::ParameterSet&);
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}
  virtual void endJob() {}
  ~JetCorrectorDBWriter() {}

 private:
  std::string inputTxtFile;
  std::string label;
  //flavour or parton option
  std::string option;
};

// Constructor
JetCorrectorDBWriter::JetCorrectorDBWriter(const edm::ParameterSet& pSet)
{
  inputTxtFile = pSet.getUntrackedParameter<std::string>("inputTxtFile");
  label        = pSet.getUntrackedParameter<std::string>("label");
  option       = pSet.getUntrackedParameter<std::string>("option");
}

// Begin Job
void JetCorrectorDBWriter::beginJob()
{
  std::string path("CondFormats/JetMETObjects/data/");
  edm::FileInPath fip(path+inputTxtFile);
  // create the parameter object from file 
  JetCorrectorParameters * payload = new JetCorrectorParameters(fip.fullPath(),option);

  // create a name for the payload 
  std::string payloadLabel(label);
  if (!option.empty())
    payloadLabel += "_"+option;

  // now write it into the DB
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) 
    {
      if (s->isNewTagRequest(payloadLabel)) 
        s->createNewIOV<JetCorrectorParameters>(payload, s->beginOfTime(), s->endOfTime(), payloadLabel);
      else 
        s->appendSinceTime<JetCorrectorParameters>(payload, 111, payloadLabel);
    }
  std::cout << "Wrote in CondDB payload label: " << payloadLabel << std::endl;
}


DEFINE_FWK_MODULE(JetCorrectorDBWriter);

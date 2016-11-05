
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/JetMETObjects/interface/MEtXYcorrectParameters.h"

namespace{
class  METCorrectorDBWriter : public edm::one::EDAnalyzer<>
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
  std::string path;
  std::string inputTxtFile;
  std::string payloadTag;
};
}

METCorrectorDBWriter::METCorrectorDBWriter(const edm::ParameterSet& pSet)
{
  era    = pSet.getUntrackedParameter<std::string>("era");
  algo   = pSet.getUntrackedParameter<std::string>("algo");
  path   = pSet.getUntrackedParameter<std::string>("path");
  payloadTag = algo;
}

void METCorrectorDBWriter::beginJob()
{
  LogDebug ("default")<<"beginJob===========";

  MEtXYcorrectParametersCollection *payload = new MEtXYcorrectParametersCollection();
  std::cout << "Starting to import payload " << payloadTag << " from text files." << std::endl;
  for( int ilev(0); ilev< MEtXYcorrectParametersCollection::N_LEVELS;++ilev)
  {
    std::string append("_");
    std::string levelName = payload->findLabel( static_cast<MEtXYcorrectParametersCollection::Level_t>(ilev) );
    //std::string levelName = MEtXYcorrectParametersCollection::findLabel( static_cast<MEtXYcorrectParametersCollection::Level_t>(ilev) );
    append += levelName;
    append += "_";
    append += algo;
    append += ".txt";
    inputTxtFile = path+era+append;
    try{
      edm::FileInPath fip(inputTxtFile);
      std::cout << "Opened file " << inputTxtFile << std::endl;
      // Create the parameter object from file
      std::vector<std::string> sections;
      payload->getSections(fip.fullPath(), sections );
      //MEtXYcorrectParametersCollection::getSections(fip.fullPath(), sections );
      if(sections.size() == 0){
        payload->push_back(ilev, MEtXYcorrectParameters(fip.fullPath(),"") );
      }else{
	for ( std::vector<std::string>::const_iterator isectbegin = sections.begin(), isectend = sections.end(), isect = isectbegin;
	      isect != isectend; ++isect ) {
	  payload->push_back( ilev, MEtXYcorrectParameters(fip.fullPath(),*isect), *isect );	  
	  std::cout << "Added level " << levelName  + "_" + *isect <<  " to record "<<ilev<< std::endl;
	}
      }
      std::cout << "Added record " << ilev << std::endl;
    }
    catch(edm::Exception ex) {
      std::cout<<"Have not found METC file: "<<inputTxtFile<<std::endl;
    }
  }

  std::cout << "Opening PoolDBOutputService" << std::endl;

  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) 
  {
    std::cout << "Setting up payload tag " << payloadTag << std::endl;
    if (s->isNewTagRequest(payloadTag))
    {
      std::cout<<"NewTagRequested"<<std::endl;
      s->createNewIOV<MEtXYcorrectParametersCollection>(payload, s->beginOfTime(), s->endOfTime(), payloadTag);
    }else{
      s->appendSinceTime<MEtXYcorrectParametersCollection>(payload, 111, payloadTag);
    }
  }
  std::cout << "Wrote in CondDB payload label: " << payloadTag << std::endl;
}


DEFINE_FWK_MODULE(METCorrectorDBWriter);


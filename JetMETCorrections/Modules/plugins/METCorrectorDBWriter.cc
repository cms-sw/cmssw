
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
#include "CondFormats/JetMETObjects/interface/METCorrectorParameters.h"

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

  METCorrectorParametersCollection *payload = new METCorrectorParametersCollection();
  std::cout << "Starting to import payload " << payloadTag << " from text files." << std::endl;
  for( int ilev(0); ilev< METCorrectorParametersCollection::N_LEVELS;++ilev)
  {
    std::string append("_");
    std::string levelName = METCorrectorParametersCollection::findLabel( static_cast<METCorrectorParametersCollection::Level_t>(ilev) );
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
      METCorrectorParametersCollection::getSections(fip.fullPath(), sections );
      if(sections.size() == 0){
        payload->push_back(ilev, METCorrectorParameters(fip.fullPath(),"") );
      }else{
	for ( std::vector<std::string>::const_iterator isectbegin = sections.begin(), isectend = sections.end(), isect = isectbegin;
	      isect != isectend; ++isect ) {
	  payload->push_back( ilev, METCorrectorParameters(fip.fullPath(),*isect), *isect );	  
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
      s->createNewIOV<METCorrectorParametersCollection>(payload, s->beginOfTime(), s->endOfTime(), payloadTag);
    }else{
      s->appendSinceTime<METCorrectorParametersCollection>(payload, 111, payloadTag);
    }
  }
  std::cout << "Wrote in CondDB payload label: " << payloadTag << std::endl;
}


DEFINE_FWK_MODULE(METCorrectorDBWriter);



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

  METCorrectorParametersCollection *payload = new METCorrectorParametersCollection();
  std::cout << "Starting to import payload " << payloadTag << " from text files." << std::endl;
  for( int i(0); i< METCorrectorParametersCollection::N_LEVELS;++i)
  {
    std::string append("_");
    std::string ilev = METCorrectorParametersCollection::findLabel( static_cast<METCorrectorParametersCollection::Level_t>(i) );
    append += ilev;
    append += "_";
    append += algo;
    append += ".txt";
    inputTxtFile = path+era+append;
    std::ifstream input( ("../../../"+inputTxtFile).c_str() );
    if ( input.good() ) {
      edm::FileInPath fip(inputTxtFile);
      std::cout << "Opened file " << inputTxtFile << std::endl;
      std::vector<std::string> sections;
      METCorrectorParametersCollection::getSections("../../../"+inputTxtFile, sections );
      if(sections.size() == 0){
        payload->push_back(i, METCorrectorParameters(fip.fullPath(),"") );
      }else{
	for ( std::vector<std::string>::const_iterator isectbegin = sections.begin(), isectend = sections.end(), isect = isectbegin;
	      isect != isectend; ++isect ) {
	  payload->push_back( i, METCorrectorParameters(fip.fullPath(),*isect), ilev + "_" + *isect );	  
	  std::cout << "Added " << ilev + "_" + *isect <<  " to record " << i << std::endl;
	}
      }
      std::cout << "Added record " << i << std::endl;
    }else{
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



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
//#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
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

// Constructor
METCorrectorDBWriter::METCorrectorDBWriter(const edm::ParameterSet& pSet)
{
  era    = pSet.getUntrackedParameter<std::string>("era");
  algo   = pSet.getUntrackedParameter<std::string>("algo");
  //payloadTag = "JetCorrectorParametersCollection_"+era+"_"+algo;
  payloadTag = algo;
}

// Begin Job
void METCorrectorDBWriter::beginJob()
{
  std::string path("CondFormats/JetMETObjects/data/");

  //JetCorrectorParametersCollection *payload = new JetCorrectorParametersCollection();
  //METCorrectorParameters *payload = new METCorrectorParameters();
   std::cout << "Starting to import payload " << payloadTag << " from text files." << std::endl;

   std::string append("_");
   append += algo;
   append += ".txt";
   inputTxtFile = path+era+append;  
   std::cout << " inputTxtFile " << inputTxtFile << std::endl;
   std::ifstream input( ("../../../"+inputTxtFile).c_str() );
   edm::FileInPath fip(inputTxtFile);
   std::string mSection = "";
   //METCorrectorParameters *payload = METCorrectorParameters(const std::string& fFile, const std::string& fSection = "");
   METCorrectorParameters *payload = new METCorrectorParameters(fip.fullPath(),mSection);
   payload->printScreen();
   if ( input.good() ) {
      edm::FileInPath fip(inputTxtFile);
      std::cout << "Opened file " << inputTxtFile << std::endl;
   }
      // create the parameter object from file 
      //std::vector<std::string> sections;
      
   
      //METCorrectorParameters *payload = METCorrectorParameters(const std::string& fFile, const std::string& fSection = "");
   /*
   /// ------------------------------------------------------- 

   for ( int i = 0; i < JetCorrectorParametersCollection::N_LEVELS; ++i ) {
    
    std::string append("_");
    std::string ilev = JetCorrectorParametersCollection::findLabel( static_cast<JetCorrectorParametersCollection::Level_t>(i) );
    append += ilev;
    append += "_";
    append += algo;
    append += ".txt"; 
    inputTxtFile = path+era+append;
    std::ifstream input( ("../../../"+inputTxtFile).c_str() );
    if ( input.good() ) {
      edm::FileInPath fip(inputTxtFile);
      std::cout << "Opened file " << inputTxtFile << std::endl;
      // create the parameter object from file 
      std::vector<std::string> sections;
      JetCorrectorParametersCollection::getSections("../../../"+inputTxtFile, sections );
      if ( sections.size() == 0 ) {
	payload->push_back( i, JetCorrectorParameters(fip.fullPath(),"") );
      }
      else {
	for ( std::vector<std::string>::const_iterator isectbegin = sections.begin(), isectend = sections.end(), isect = isectbegin;
	      isect != isectend; ++isect ) {
	  payload->push_back( i, JetCorrectorParameters(fip.fullPath(),*isect), ilev + "_" + *isect );	  
	  std::cout << "Added " << ilev + "_" + *isect <<  " to record " << i << std::endl;
	}
      }
      std::cout << "Added record " << i << std::endl;
    } else {
      std::cout << "Did not find JEC file " << inputTxtFile << std::endl;
    }
    
  }
  */  
  std::cout << "Opening PoolDBOutputService" << std::endl;

  // now write it into the DB
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


DEFINE_FWK_MODULE(METCorrectorDBWriter);


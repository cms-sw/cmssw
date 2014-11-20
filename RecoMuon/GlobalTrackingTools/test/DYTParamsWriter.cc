#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/RecoMuonObjects/interface/DYTParamsObject.h"
#include "CondFormats/DataRecord/interface/DYTParamsObjectRcd.h" 
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <string>
#include <fstream>
#include <iostream>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string_regex.hpp>
#include <boost/regex.hpp>

class DYTParamsWriter : public edm::EDAnalyzer 
{

public:

  explicit DYTParamsWriter(const edm::ParameterSet&);
  ~DYTParamsWriter();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  virtual void   endRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void   analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void   endJob() override;

private:

  void parseLine(std::string & line, uint32_t & id, std::vector<double> & pars);
  int getNParamsFromFunction();
  
  const std::string m_inputFileName;
  const std::string m_inputFunction;

  edm::Service<cond::service::PoolDBOutputService> m_poolDbService;
  
  DYTParamsObject * m_params;

};


DYTParamsWriter::DYTParamsWriter(const edm::ParameterSet& iConfig) :
  m_inputFileName(iConfig.getParameter<std::string>("inputFileName")),
  m_inputFunction(iConfig.getParameter<std::string>("inputFunction"))	      
{

  if( !m_poolDbService.isAvailable() )
    throw cms::Exception("NotAvailable") << "[DYTParamsWriter::DYTParamsWriter] "
					 << "PoolDBOutputService is not available.";
    
  if (!m_poolDbService->isNewTagRequest("DYTParamsObjectRcd")) 
    throw cms::Exception("ObjectExists") << "[DYTParamsWriter::DYTParamsWriter] "
					 << "The output file already contains a valid "
					 << "\"DYTParamsObjectRcd\" record." << std::endl
					 << "Please provide a different file name or tag.";

  m_params = new DYTParamsObject;

}


DYTParamsWriter::~DYTParamsWriter()
{

}


void DYTParamsWriter::analyze(edm::Event const&, edm::EventSetup const&)
{

}


void DYTParamsWriter::endRun(edm::Run const&, edm::EventSetup const&)
{

  std::ifstream inputFile(m_inputFileName.c_str());

  int nLines=0;
  uint32_t nParams=getNParamsFromFunction();

  m_params->setFormula(m_inputFunction);

  std::string line;

  while(std::getline(inputFile, line)) 
    {
      
      nLines++;

      uint32_t id;
      std::vector<double> params;

      parseLine(line, id, params);

      if ( nParams != params.size() )
	throw cms::Exception("ConfigFileError") << "[DYTParamsWriter::endRun] "
						<< "# of parameters in line " << nLines
						<< " differs from the one expected from"
						<< " the parametrized function (" << nParams << ").";
      
      DYTParamObject obj(id, params);
      m_params->addParamObject(obj);      
      
    }
  
  std::cout << "[DYTParamsWriter::endRun] " 
	    << "Processed " << nLines << " lines with " 
	    << nParams << " parameters for each line." << std::endl;
  
  inputFile.close();
  
}


void DYTParamsWriter::endJob() 
{

  // Writing to DB
  if( m_poolDbService.isAvailable() )
    m_poolDbService->writeOne(m_params, m_poolDbService->beginOfTime(), "DYTParamsObjectRcd"); 
  else 
    throw cms::Exception("NotAvailable") << "[DYTParamsWriter::endJob] "
					 << "PoolDBOutputService is not available.";
}


void DYTParamsWriter::parseLine(std::string & line, uint32_t & id, std::vector<double> & pars)
{

  std::vector<std::string> elements;
  boost::algorithm::split(elements,line,boost::algorithm::is_any_of(" \t\n"));
  if (elements.size() < 2) 
    {
      throw cms::Exception("ConfigFileError") << "[DYTParamsWriter::parseLine] "
					      << "Wrong number of entries in line : \"" 
					      << line << "\" ." << std::endl;
    } 
  else
    {
      id = atoi(elements[0].c_str());
      for (uint32_t i=1; i < elements.size(); ++i)
	pars.push_back(atof(elements[i].c_str()));
    }

}


int DYTParamsWriter::getNParamsFromFunction()
{

  std::vector<std::string> parameters;
  boost::regex reg("\\[[0-9]+\\]");

  boost::sregex_token_iterator rIt(m_inputFunction.begin(), m_inputFunction.end(), reg, 0);
  boost::sregex_token_iterator rEnd;

  int nParams = 0;
  for( ; rIt != rEnd; ++rIt )
    nParams++;

  return nParams;

}


void DYTParamsWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{

  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("inputFileName", "dyt_params.txt");
  desc.add<std::string>("inputFunction", "[0]*x + [1]*x^[2]");
  descriptions.add("dytParamsWriter",desc);

}

DEFINE_FWK_MODULE(DYTParamsWriter);

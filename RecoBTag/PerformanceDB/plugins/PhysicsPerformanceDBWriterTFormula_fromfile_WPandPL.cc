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
//#include "CondFormats/PhysicsPerformance/interface/PhysicsPerformancePayload.h"
//#include "CondFormats/DataRecord/interface/PerformancePayloadRecord.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"

class PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL : public edm::EDAnalyzer
{
public:
  PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL(const edm::ParameterSet&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}
  virtual void endJob() {}
  ~PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL() {}

private:
  std::string inputTxtFile;
  std::string rec1,rec2;
  
};

PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL::PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL
  (const edm::ParameterSet& p)
{
  inputTxtFile = p.getUntrackedParameter<std::string>("inputTxtFile");
  rec1 = p.getUntrackedParameter<std::string>("RecordPayload");
  rec2 = p.getUntrackedParameter<std::string>("RecordWP");
}

void PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL::beginJob(const edm::EventSetup&)
{
  //
  // read object from file
  //

  //
  // File Format is
  // - tagger name
  // - cut
  // - concrete class name
  // number of results (== number of formulas)
  // number of variables in the parameterization
  // - results (as ints)
  // - variables (as ints)
  // - formulas
  // - the limits
  //

  std::ifstream in;
  in.open(inputTxtFile.c_str());
  std::string tagger;
  float cut;
 
  std::string concreteType;
  std::vector< std::pair<float, float> > limits;
  std::vector<std::string> formulas;

  in >> tagger;
  std::cout << "WP Tagger is "<<tagger<<std::endl;
  
  in >> cut;
  std::cout << "WP Cut is "<<cut<<std::endl;

  in >> concreteType;
  std::cout << "concrete Type is "<<concreteType<<std::endl;

  int nres=0, nvar=0;
  
  in >> nres;
  in >> nvar;

  std::cout <<" Using "<<nres<<" results and "<< nvar<<" variables"<<std::endl;

  int number=0;;

  std::vector<PerformanceResult::ResultType> res;
  std::vector<BinningVariables::BinningVariablesType> bin;
  //
  // read results
  //
  number=0;
  while (number<nres && !in.eof()) {
    int tmp;
    in>> tmp;
    res.push_back((PerformanceResult::ResultType)(tmp));
    std::cout <<" Result #"<<number <<" is "<<tmp<<std::endl;;
    number++;
  }
  if (number != nres){
    std::cout <<" Table not well formed"<<std::endl;
  }

  //
  // read the variables
  //


  number=0;
  while (number<nvar && !in.eof()) {
    int tmp;
    in>> tmp;
    bin.push_back((BinningVariables::BinningVariablesType)(tmp));
    std::cout <<" Variable #"<<number <<" is "<<tmp<<std::endl;;
    number++;
  }
  if (number != nvar){
    std::cout <<" Table not well formed"<<std::endl;
  }


  //
  // now read the formulas
  //
  number =0;

  while (number < nres && (!in.eof())){
    std::string temp;
    in >> temp;
    std::cout <<" Inserting "<<temp<< " as formula in position "<<number<<std::endl;
    number++;
    formulas.push_back(temp);
  }

  if (nres!= number ){
    std::cout <<" NOT OK, this is not what I would expect"<<std::endl;
    abort();
  }


  number=0;
  while (number < nvar && (!in.eof())){
    float temp1,temp2;
    in >> temp1;
    in >> temp2;
    std::cout <<" Inserting "<<temp1<<","<<temp2<< " as limits in position "<<number<<std::endl;
    number++;
    limits.push_back(std::pair<float, float>(temp1,temp2));
  }
  if (nvar != number ){
    std::cout <<" NOT OK, this is not what I would expect"<<std::endl;
    abort();
  }

  in.close();


  PerformanceWorkingPoint * wp = new PerformanceWorkingPoint(cut, tagger);
  PerformancePayloadFromTFormula * btagpl = 0;

  PhysicsTFormulaPayload ppl(limits, formulas);


  if (concreteType == "PerformancePayloadFromTFormula"){
    btagpl = new PerformancePayloadFromTFormula(res, bin, ppl);
  }else{
    std::cout <<" Non existing request: " <<concreteType<<std::endl;
  }
  
  std::cout <<" Created the "<<concreteType <<" object"<<std::endl;
  
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable())
    {
      if (s->isNewTagRequest(rec1))
	{
	  s->createNewIOV<PerformancePayload>(btagpl,
						  s->beginOfTime(),
						  s->endOfTime(),
						  rec1);
	}
      else
	{
	  
	  s->appendSinceTime<PerformancePayload>(btagpl,
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
	  s->createNewIOV<PerformanceWorkingPoint>(wp,
					    s->beginOfTime(),
					    s->endOfTime(),
					    rec2);
	}
      else
	{
	  
	  s->appendSinceTime<PerformanceWorkingPoint>(wp,
					       /// JUST A STUPID PATCH
					       111,
					       rec2);
	}
    }
  
  
  
}




DEFINE_FWK_MODULE(PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL);

#include "GeneratorInterface/Core/interface/GenXSecAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

GenXSecAnalyzer::GenXSecAnalyzer(const edm::ParameterSet& iConfig):
  hepidwtup_(-1),
  xsec_(0,0)
{
  products_.clear();
}

GenXSecAnalyzer::~GenXSecAnalyzer()
{
}

void
GenXSecAnalyzer::beginJob() {
  products_.clear();  
}

void
GenXSecAnalyzer::analyze(const edm::Event&, const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------

void
GenXSecAnalyzer::endLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {

  edm::Handle<GenLumiInfoProduct> genLumiInfo;
  iLumi.getByLabel("generator",genLumiInfo);

  const GenLumiInfoProduct one = *(genLumiInfo);
  sampleInfo thisProcessInfos = one.getProcessInfos();
  hepidwtup_ = one.getHEPIDWTUP();

  // if it's a pure parton-shower generator, check there should be only one element in thisProcessInfos
  // the error of lheXSec is -1
  if(hepidwtup_== -1)
    {
      if(thisProcessInfos.size()!=1){
	edm::LogError("GenXSecAnalyzer::endLuminosityBlock") << "Pure parton shower has thisProcessInfos size!=1";
	return;
      }
      if(thisProcessInfos[0].lheXSec().value()<1e-6){
	edm::LogError("GenXSecAnalyzer::endLuminosityBlock") << "cross section value = "  << thisProcessInfos[0].lheXSec().value();
	return;
      }
    }

  // now determine if this LuminosityBlock has the same lheXSec as existing products
  bool sameMC = false;
  for(unsigned int i=0; i < products_.size(); i++){

    unsigned int nProcesses = products_[i].size();
    if(nProcesses != thisProcessInfos.size())continue;
    bool isOK=true;
    for(unsigned int ip=0; ip < nProcesses; ip++){
      if(products_[i][ip].process() != thisProcessInfos[ip].process()){isOK=false;break;}
      if(products_[i][ip].lheXSec() != thisProcessInfos[ip].lheXSec()){isOK=false; break;}
    }
    if(isOK){
      sameMC = true;
      for(unsigned int ip=0; ip < nProcesses; ip++)
	products_[i][ip].addOthers(thisProcessInfos[ip]);
    }
  }
  
  if(!sameMC)
    products_.push_back(thisProcessInfos);
  return;
}

void 
GenXSecAnalyzer::compute()
{
  // for pure parton shower generator
  
  if(hepidwtup_== -1)
    {
      double sigSum = 0.0;
      double totalN = 0.0;
      for(unsigned int i=0; i < products_.size(); i++){

	GenLumiInfoProduct::ProcessInfo proc = products_[i][0];	  
	double hepxsec_value = proc.lheXSec().value();

	sigSum += proc.tried().sum() * hepxsec_value;
	totalN += proc.tried().sum();
      }
      double sigAve = totalN>1e-6? sigSum/totalN: 0;
      xsec_ = GenLumiInfoProduct::XSec(sigAve,-1);      
    }
  // for ME+parton shower MC
  else{

    double sum_numerator = 0;
    double sum_denominator = 0;
  
    for(unsigned int i=0; i < products_.size(); i++){

      double sigSelSum = 0.0;
      double sigSum = 0.0;
      double sigBrSum = 0.0;
      double err2Sum = 0.0;
      double errBr2Sum = 0.0;

      for(unsigned int ip=0; ip < products_[i].size(); ip++){
	GenLumiInfoProduct::ProcessInfo proc = products_[i][ip];	  
	double hepxsec_value = proc.lheXSec().value();
	double hepxsec_error = proc.lheXSec().error();

	double sigmaSum, sigma2Sum, sigma2Err;
	sigmaSum = proc.tried().sum() * hepxsec_value;
	sigma2Sum = proc.tried().sum2() * hepxsec_value * hepxsec_value;
	sigma2Err = proc.tried().sum2() * hepxsec_error * hepxsec_error;
      
	if (!proc.killed().n())
	  continue;
    
	double sigmaAvg = sigmaSum / proc.tried().sum();
	double fracAcc = proc.killed().sum() / proc.selected().sum();
	double fracBr = proc.accepted().sum() > 0.0 ?
	  proc.acceptedBr().sum() / proc.accepted().sum() : 1;
	double sigmaFin = sigmaAvg * fracAcc * fracBr;
	double sigmaFinBr = sigmaFin * fracBr;
      
	double relErr = 1.0;
	if (proc.killed().n() > 1) {
	  double sigmaAvg2 = sigmaAvg * sigmaAvg;
	  double delta2Sig =
	    (sigma2Sum / proc.tried().n() - sigmaAvg2) /
	    (proc.tried().n() * sigmaAvg2);
	  double delta2Veto =
	    ((double)proc.selected().n() - proc.killed().n()) /
	    ((double)proc.selected().n() * proc.killed().n());
	  double delta2Sum = delta2Sig + delta2Veto
	    + sigma2Err / sigma2Sum;
	  relErr = (delta2Sum > 0.0 ?
		    std::sqrt(delta2Sum) : 0.0);
	}
	double deltaFin = sigmaFin * relErr;
	double deltaFinBr = sigmaFinBr * relErr;

	sigSelSum += sigmaAvg;
	sigSum += sigmaFin;
	sigBrSum += sigmaFinBr;
	err2Sum += deltaFin * deltaFin;
	errBr2Sum += deltaFinBr * deltaFinBr;
      } // end of loop over different processes
      
      double dN = std::sqrt(errBr2Sum);
      sum_denominator  +=  (dN> 1e-6)? 1/dN/dN: 0;
      sum_numerator    +=  (dN> 1e-6)? sigBrSum/dN/dN: 0;


    } // end of loop over different samples
    double final_value = sum_denominator > 1e-6? sum_numerator/sum_denominator : 0;
    double final_error = sum_denominator > 1e-6? 1/sqrt(sum_denominator) : -1;
    xsec_ = GenLumiInfoProduct::XSec(final_value, final_error);
  }
  return;
}


void
GenXSecAnalyzer::endJob() {

  if(products_.size()>0)
    compute();

  std::cout << "Final Xsec = " << xsec_.value() << " +- " << xsec_.error() << std::endl;

}


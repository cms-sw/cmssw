#include "GeneratorInterface/Core/interface/GenXSecAnalyzer.h"
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
  std::vector<GenLumiInfoProduct::ProcessInfo> thisProcessInfos = one.getProcessInfos();
  hepidwtup_ = one.getHEPIDWTUP();

  bool isOK=true;
  if(products_.size()==0){
    for(unsigned int i=0; i < thisProcessInfos.size(); i++){
      products_.push_back(thisProcessInfos[i]);
      
    }
  }
  else
    for(unsigned int i=0; i < products_.size(); i++){
      if(products_[i].process() != thisProcessInfos[i].process())isOK=false;
      if(products_[i].lheXSec() != thisProcessInfos[i].lheXSec())isOK=false;
      if(!isOK)break;
      products_[i].addOthers(thisProcessInfos[i]);
  }
  return;
}

void 
GenXSecAnalyzer::compute()
{

  double sigSelSum = 0.0;
  double sigSum = 0.0;
  double sigBrSum = 0.0;
  double err2Sum = 0.0;
  double errBr2Sum = 0.0;
  
  for(unsigned int ip=0; ip < products_.size(); ip++){

    GenLumiInfoProduct::ProcessInfo proc = products_[ip];	  
    double hepxsec_value = proc.lheXSec().value();
    double hepxsec_error = proc.lheXSec().error();

    double sigmaSum, sigma2Sum, sigma2Err, xsec;
    switch(std::abs(hepidwtup_)) {
    case 2:
      sigmaSum = proc.tried().sum() * hepxsec_value;
      sigma2Sum = proc.tried().sum2() * hepxsec_value * hepxsec_value;
      sigma2Err = proc.tried().sum2() * hepxsec_error * hepxsec_error;
      break;
    case 3:
      sigmaSum = proc.tried().n() * hepxsec_value;
      sigma2Sum = sigmaSum * hepxsec_value;
      sigma2Err = proc.tried().n() * hepxsec_error * hepxsec_error;
      break;
    default:
      xsec = proc.tried().sum() / proc.tried().n();
      sigmaSum = proc.tried().sum() * xsec;
      sigma2Sum = proc.tried().sum2() * xsec * xsec;
      sigma2Err = 0.0;
    }

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
  }

  xsec_ = GenLumiInfoProduct::XSec(sigBrSum,std::sqrt(errBr2Sum));

}


void
GenXSecAnalyzer::endJob() {

  if(products_.size()>0)
    compute();

  std::cout << "Final Xsec = " << xsec_.value() << " +- " << xsec_.error() << std::endl;

}


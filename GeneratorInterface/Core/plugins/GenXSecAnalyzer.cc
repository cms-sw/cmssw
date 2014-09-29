#include "GeneratorInterface/Core/interface/GenXSecAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

GenXSecAnalyzer::GenXSecAnalyzer(const edm::ParameterSet& iConfig):
  hepidwtup_(-1),
  xsec_(0,0),
  jetMatchEffStat_(0,0,0,0,0.,0.,0.,0.),
  totalEffStat_(0,0,0,0,0.,0.,0.,0.)
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

  edm::Handle<GenFilterInfo> genFilter;
  if(iLumi.getByLabel("genFilterEfficiencyProducer", genFilter))
    totalEffStat_.mergeProduct(*genFilter);


  edm::Handle<GenLumiInfoProduct> genLumiInfo;
  iLumi.getByLabel("generator",genLumiInfo);

  hepidwtup_ = genLumiInfo->getHEPIDWTUP();

  std::vector<GenLumiInfoProduct::ProcessInfo> theProcesses = genLumiInfo->getProcessInfos();
  unsigned int theProcesses_size = theProcesses.size();
  // if it's a pure parton-shower generator, check there should be only one element in thisProcessInfos
  // the error of lheXSec is -1
  if(hepidwtup_== -1)
    {
      if(theProcesses_size!=1){
	edm::LogError("GenXSecAnalyzer::endLuminosityBlock") << "Pure parton shower has thisProcessInfos size!=1";
	return;
      }
      if(theProcesses[0].lheXSec().value()<1e-6){
	edm::LogError("GenXSecAnalyzer::endLuminosityBlock") << "cross section value = "  << theProcesses[0].lheXSec().value();
	return;
      }
    }

  // doing generic summing 
  for(unsigned int ip=0; ip < theProcesses_size; ip++)
    {
      GenLumiInfoProduct::FinalStat temp_killed   = theProcesses[ip].killed();
      GenLumiInfoProduct::FinalStat temp_selected = theProcesses[ip].selected();
      double passw  = temp_killed.sum();
      double passw2 = temp_killed.sum2();
      double totalw  = temp_selected.sum();
      double totalw2 = temp_selected.sum2();
      jetMatchEffStat_.mergeProduct(GenFilterInfo(
						  theProcesses[ip].nPassPos(),
						  theProcesses[ip].nPassNeg(),
						  theProcesses[ip].nTotalPos(),
						  theProcesses[ip].nTotalNeg(),
						  passw,
						  passw2,
						  totalw,
						  totalw2)
				    );

    }
  // now determine if this LuminosityBlock has the same lheXSec as existing products
  bool sameMC = false;
  for(unsigned int i=0; i < products_.size(); i++){

    if(products_[i].mergeProduct(*genLumiInfo))
      sameMC = true;
    else if(!products_[i].samePhysics(*genLumiInfo))
      {
	edm::LogError("GenXSecAnalyzer::endLuminosityBlock") << "Merging samples that come from different physics processes";
	return;
      }

  }
  
  if(!sameMC)
      products_.push_back(*genLumiInfo);
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

	GenLumiInfoProduct::ProcessInfo proc = products_[i].getProcessInfos()[0];	  
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

      unsigned int vectorSize = products_[i].getProcessInfos().size();
      for(unsigned int ip=0; ip < vectorSize; ip++){
	GenLumiInfoProduct::ProcessInfo proc = products_[i].getProcessInfos()[ip];	  
	double hepxsec_value = proc.lheXSec().value();
	double hepxsec_error = proc.lheXSec().error();

	if (!proc.killed().n())
	  continue;

	double sigma2Sum, sigma2Err;
	sigma2Sum = hepxsec_value * hepxsec_value;
	sigma2Err = hepxsec_error * hepxsec_error;

	double sigmaAvg = hepxsec_value;

	double fracAcc = 0;
	double ntotal = proc.nTotalPos()-proc.nTotalNeg();
	double npass  = proc.nPassPos() -proc.nPassNeg();
	switch(hepidwtup_){
	case 3: case -3:
	  fracAcc = ntotal > 1e-6? npass/ntotal: -1;
	    break;
	default:
	  fracAcc = proc.selected().sum() > 1e-6? proc.killed().sum() / proc.selected().sum():-1;
	  break;
	}

	if(fracAcc<1e-6)continue;

	double fracBr = proc.accepted().sum() > 0.0 ?
	  proc.acceptedBr().sum() / proc.accepted().sum() : 1;
	double sigmaFin = sigmaAvg * fracAcc * fracBr;
	double sigmaFinBr = sigmaFin * fracBr;

	double relErr = 1.0;
	if (proc.killed().n() > 1) {
	  double efferr2=0;
	  switch(hepidwtup_) {
	  case 3: case -3:
	    {
	      double ntotal_pos = proc.nTotalPos();
	      double effp  = ntotal_pos > 1e-6?
		(double)proc.nPassPos()/ntotal_pos:0;
	      double effp_err2 = ntotal_pos > 1e-6?
		(1-effp)*effp/ntotal_pos: 0;

	      double ntotal_neg = proc.nTotalNeg();
	      double effn  = ntotal_neg > 1e-6?
		(double)proc.nPassNeg()/ntotal_neg:0;
	      double effn_err2 = ntotal_neg > 1e-6?
		(1-effn)*effn/ntotal_neg: 0;

	      efferr2 = ntotal > 0 ? 
		(ntotal_pos*ntotal_pos*effp_err2 +
		 ntotal_neg*ntotal_neg*effn_err2)/ntotal/ntotal:0;
	      break;
	    }
	  default:
	    {
	      double denominator = pow(proc.selected().sum(),4);
	      double passw       = proc.killed().sum();
	      double passw2      = proc.killed().sum2();
	      double failw       = proc.selected().sum() - passw;
	      double failw2      = proc.selected().sum2() - passw2;
	      double numerator   = (passw2*failw*failw + failw2*passw*passw); 
			    
	      efferr2 = denominator>1e-6?
		numerator/denominator:0;
	      break;
	    }
	  }
	  double delta2Veto = efferr2/fracAcc/fracAcc;
	  double delta2Sum = delta2Veto
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
    double final_value = sum_denominator > 0? sum_numerator/sum_denominator : 0;
    double final_error = sum_denominator > 0? 1/sqrt(sum_denominator) : -1;
    xsec_ = GenLumiInfoProduct::XSec(final_value, final_error);
  }
  return;
}


void
GenXSecAnalyzer::endJob() {

  if(products_.size()>0)
    compute();

  double filterOnly_eff = totalEffStat_.filterEfficiency(hepidwtup_);
  double filterOnly_err = totalEffStat_.filterEfficiencyError(hepidwtup_);
  
  double jetmatching_eff_total = jetMatchEffStat_.filterEfficiency(hepidwtup_);
  double jetmatching_err_total = jetMatchEffStat_.filterEfficiencyError(hepidwtup_);

  std::cout << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << "Overall cross-section summary:" << std::endl;
  std::cout << "------------------------------------" << std::endl;
  
  std::cout << "jet matching efficiency = " << 
    jetMatchEffStat_.sumPassWeights() << "/" << 
    jetMatchEffStat_.sumWeights() << " = " 
	   << std::setprecision(6) << jetmatching_eff_total << " +- " << jetmatching_err_total << std::endl;

  std::cout << "Before filter: cross section = " << std::setprecision(6)  << xsec_.value() << " +- " << std::setprecision(6) << xsec_.error() <<  " pb" << std::endl;

  std::cout << "Filter efficiency = " << std::setprecision(6)  
	    << filterOnly_eff << " +- " << filterOnly_err << std::endl;


  double xsec_after  = xsec_.value()*filterOnly_eff ;
  double error_after = xsec_after*sqrt(xsec_.error()*xsec_.error()/xsec_.value()/xsec_.value()+
				  filterOnly_err*filterOnly_err/filterOnly_eff/filterOnly_eff);

  std::cout << "After filter: cross section = " 
	    << std::setprecision(6) << xsec_after
	    << " +- " 
	    << std::setprecision(6) << error_after
	    << " pb"
	    << std::endl;

}


#include "GeneratorInterface/Core/interface/GenXSecAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

GenXSecAnalyzer::GenXSecAnalyzer(const edm::ParameterSet& iConfig):
  hepidwtup_(-1),
  theProcesses_size(0),
  hasHepMCFilterInfo_(false),
  xsec_(0,0),
  filterOnlyEffStat_(0,0,0,0,0.,0.,0.,0.),
  hepMCFilterEffStat_(0,0,0,0,0.,0.,0.,0.)
{
  eventEffStat_.clear();
  jetMatchEffStat_.clear();
  xsecBeforeMatching_.clear();
  xsecAfterMatching_.clear();
  products_.clear();
  genFilterInfoToken_ = consumes<GenFilterInfo,edm::InLumi>(edm::InputTag("genFilterEfficiencyProducer",""));
  hepMCFilterInfoToken_ = consumes<GenFilterInfo,edm::InLumi>(edm::InputTag("generator",""));
  genLumiInfoToken_ = consumes<GenLumiInfoProduct,edm::InLumi>(edm::InputTag("generator",""));
}

GenXSecAnalyzer::~GenXSecAnalyzer()
{
}

void
GenXSecAnalyzer::beginJob() {
  eventEffStat_.clear();
  jetMatchEffStat_.clear();
  xsecBeforeMatching_.clear();
  xsecAfterMatching_.clear();
  products_.clear();  
}

void
GenXSecAnalyzer::analyze(const edm::Event&, const edm::EventSetup&)
{
}


void
GenXSecAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {
}

void
GenXSecAnalyzer::endLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {
  // add up information of GenFilterInfo from different luminosity blocks
  edm::Handle<GenFilterInfo> genFilter;
  iLumi.getByToken(genFilterInfoToken_,genFilter);
  if(genFilter.isValid())
    filterOnlyEffStat_.mergeProduct(*genFilter);


  edm::Handle<GenFilterInfo> hepMCFilter;
  iLumi.getByToken(hepMCFilterInfoToken_,hepMCFilter);
  hasHepMCFilterInfo_ = hepMCFilter.isValid();
  if(hasHepMCFilterInfo_)
    hepMCFilterEffStat_.mergeProduct(*hepMCFilter);


  edm::Handle<GenLumiInfoProduct> genLumiInfo;
  iLumi.getByToken(genLumiInfoToken_,genLumiInfo);
  if (!genLumiInfo.isValid()) return;

  hepidwtup_ = genLumiInfo->getHEPIDWTUP();

  std::vector<GenLumiInfoProduct::ProcessInfo> theProcesses = genLumiInfo->getProcessInfos();
  theProcesses_size = theProcesses.size();
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


  // initialize jetMatchEffStat with nprocess+1 elements
  if(jetMatchEffStat_.size()==0){
    
    for(unsigned int ip=0; ip < theProcesses_size; ip++)
      {
	jetMatchEffStat_.push_back(GenFilterInfo(0,0,0,0,0.,0.,0.,0.));
      }
    jetMatchEffStat_.push_back(GenFilterInfo(0,0,0,0,0.,0.,0.,0.));
  }

  // initialize event-level statistics with nprocess+1 elements
  if(eventEffStat_.size()==0){
    
    for(unsigned int ip=0; ip < theProcesses_size; ip++)
      {
	eventEffStat_.push_back(GenFilterInfo(0,0,0,0,0.,0.,0.,0.));
      }
    eventEffStat_.push_back(GenFilterInfo(0,0,0,0,0.,0.,0.,0.));
  }

  // doing generic summing for jet matching statistics
  for(unsigned int ip=0; ip < theProcesses_size; ip++)
    {
      GenLumiInfoProduct::FinalStat temp_killed   = theProcesses[ip].killed();
      GenLumiInfoProduct::FinalStat temp_selected = theProcesses[ip].selected();
      double passw  = temp_killed.sum();
      double passw2 = temp_killed.sum2();
      double totalw  = temp_selected.sum();
      double totalw2 = temp_selected.sum2();
      // matching statistics for each process
      jetMatchEffStat_[ip].mergeProduct(GenFilterInfo(
						      theProcesses[ip].nPassPos(),
						      theProcesses[ip].nPassNeg(),
						      theProcesses[ip].nTotalPos(),
						      theProcesses[ip].nTotalNeg(),
						      passw,
						      passw2,
						      totalw,
						      totalw2)
					);
      // matching statistics for all processes
      jetMatchEffStat_[theProcesses_size].mergeProduct(GenFilterInfo(
								     theProcesses[ip].nPassPos(),
								     theProcesses[ip].nPassNeg(),
								     theProcesses[ip].nTotalPos(),
								     theProcesses[ip].nTotalNeg(),
								     passw,
								     passw2,
								     totalw,
								     totalw2)
						       );



      // event-level statistics for each process
      eventEffStat_[ip].mergeProduct(GenFilterInfo(
						   theProcesses[ip].nPassPos()+theProcesses[ip].nPassNeg(),
						   0,
						   theProcesses[ip].nTotalPos()+theProcesses[ip].nTotalNeg(),
						   0,
						   theProcesses[ip].nPassPos()+theProcesses[ip].nPassNeg(),
						   theProcesses[ip].nPassPos()+theProcesses[ip].nPassNeg(),
						   theProcesses[ip].nTotalPos()+theProcesses[ip].nTotalNeg(),
						   theProcesses[ip].nTotalPos()+theProcesses[ip].nTotalNeg())
				     );
      // event-level statistics for all processes
      eventEffStat_[theProcesses_size].mergeProduct(GenFilterInfo(
								  theProcesses[ip].nPassPos()+theProcesses[ip].nPassNeg(),
								  0,
								  theProcesses[ip].nTotalPos()+theProcesses[ip].nTotalNeg(),
								  0,
								  theProcesses[ip].nPassPos()+theProcesses[ip].nPassNeg(),
								  theProcesses[ip].nPassPos()+theProcesses[ip].nPassNeg(),
								  theProcesses[ip].nTotalPos()+theProcesses[ip].nTotalNeg(),
								  theProcesses[ip].nTotalPos()+theProcesses[ip].nTotalNeg())
						    );
     

    }


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
      // average over number of events since the cross sections have no errors
      double sigAve = totalN>1e-6? sigSum/totalN: 0;
      xsecBeforeMatching_.push_back(GenLumiInfoProduct::XSec(sigAve,-1));
      xsecAfterMatching_.push_back(GenLumiInfoProduct::XSec(sigAve,-1));
    }
  // for ME+parton shower MC
  else{

    const unsigned int sizeOfInfos= theProcesses_size+1;
    // for computing cross sectiona before matching
    double sum_numerator_before[sizeOfInfos]; 
    double sum_denominator_before[sizeOfInfos];

    // for computing cross sectiona after matching
    double sum_numerator_after[sizeOfInfos];
    double sum_denominator_after[sizeOfInfos];

    // initialize every element with zero
    for(unsigned int i=0; i < sizeOfInfos; i++)
      {
	sum_numerator_before[i]=0; 
	sum_denominator_before[i]=0;
	sum_numerator_after[i]=0;
	sum_denominator_after[i]=0;
      }
  
    // loop over different MC samples
    for(unsigned int i=0; i < products_.size(); i++){

      // sum of cross sections and errors over different processes
      double sigSelSum = 0.0;
      double errSel2Sum = 0.0;
      double sigSum = 0.0;
      double err2Sum = 0.0;

      // loop over different processes for each sample
      unsigned int vectorSize = products_[i].getProcessInfos().size();
      for(unsigned int ip=0; ip < vectorSize; ip++){
	GenLumiInfoProduct::ProcessInfo proc = products_[i].getProcessInfos()[ip];	  
	double hepxsec_value = proc.lheXSec().value();
	double hepxsec_error = proc.lheXSec().error();

	// skips computation if jet matching efficiency=0
	if (proc.killed().n()<1)
	  continue;

	// computing jet matching efficiency for this process
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

	// cross section after matching for this particular process
	double sigmaFin = hepxsec_value * fracAcc;

	// computing error on jet matching efficiency
	double relErr = 1.0;
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
	
	// computing total error on cross section after matching efficiency

	double sigma2Sum, sigma2Err;
	sigma2Sum = hepxsec_value * hepxsec_value;
	sigma2Err = hepxsec_error * hepxsec_error;

	// sum of cross sections before matching and errors over different samples for each process
	sum_denominator_before[ip]  +=  (sigma2Err > 0)? 1/sigma2Err: 0;
	sum_numerator_before[ip]    +=  (sigma2Err > 0)?  hepxsec_value/sigma2Err: 0;


	double delta2Sum = delta2Veto
	  + sigma2Err / sigma2Sum;
	relErr = (delta2Sum > 0.0 ?
		  std::sqrt(delta2Sum) : 0.0);
	double deltaFin = sigmaFin * relErr;

	// sum of cross sections and errors over different processes
	sigSelSum += hepxsec_value;
	errSel2Sum += sigma2Err;
	sigSum += sigmaFin;
	err2Sum += deltaFin * deltaFin;
	
	// sum of cross sections after matching and errors over different samples for each process
	sum_denominator_after[ip]  +=  (deltaFin > 0)? 1/(deltaFin * deltaFin): 0;
	sum_numerator_after[ip]    +=  (deltaFin > 0)? sigmaFin/(deltaFin * deltaFin): 0;


      } // end of loop over different processes

      sum_denominator_before[sizeOfInfos-1]  +=  (errSel2Sum> 0)? 1/errSel2Sum: 0;
      sum_numerator_before[sizeOfInfos-1]    +=  (errSel2Sum> 0)? sigSelSum/errSel2Sum: 0;
      
      sum_denominator_after[sizeOfInfos-1]  +=  (err2Sum>0)? 1/err2Sum: 0;
      sum_numerator_after[sizeOfInfos-1]    +=  (err2Sum>0)? sigSum/err2Sum: 0;


    } // end of loop over different samples
  
    
    for(unsigned int i=0; i<sizeOfInfos; i++)
      {
	double final_value = (sum_denominator_before[i]>0) ? (sum_numerator_before[i]/sum_denominator_before[i]):0;
	double final_error = (sum_denominator_before[i]>0) ? (1/sqrt(sum_denominator_before[i])):-1;
	xsecBeforeMatching_.push_back(GenLumiInfoProduct::XSec(final_value, final_error));
	
	double final_value2 = (sum_denominator_after[i]>0) ? (sum_numerator_after[i]/sum_denominator_after[i]):0;
        double final_error2 = (sum_denominator_after[i]>0) ? 1/sqrt(sum_denominator_after[i]):-1;
        xsecAfterMatching_.push_back(GenLumiInfoProduct::XSec(final_value2, final_error2));
      }
  	
  }
  return;
}


void
GenXSecAnalyzer::endJob() {

  edm::LogPrint("GenXSecAnalyzer") << "\n"
  << "------------------------------------" << "\n"
  << "GenXsecAnalyzer:" << "\n"
  << "------------------------------------";
  
  if(!products_.size()) {
    edm::LogPrint("GenXSecAnalyzer") << "------------------------------------" << "\n"
    << "Cross-section summary not available" << "\n"
    << "------------------------------------";
    return;
  }
  
  
  compute();


  edm::LogPrint("GenXSecAnalyzer") 
    << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- \n"
    << "Overall cross-section summary:" << "\n"
    << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------";
  edm::LogPrint("GenXSecAnalyzer")  
    << "Process\t\txsec_before [pb]\t\tpassed\tnposw\tnnegw\ttried\tnposw\tnnegw \txsec_match [pb]\t\t\taccepted [%]\t event_eff [%]";

  const int sizeOfInfos= theProcesses_size+1;
  const int last = sizeOfInfos-1;
  std::string * title = new std::string[sizeOfInfos];

  for(int i=0; i < sizeOfInfos; i++){

    double jetmatch_eff=0;
    double jetmatch_err=0;

    if(i==last)
      {
	title[i] = "Total";

	edm::LogPrint("GenXSecAnalyzer") 
	  << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ";
	jetmatch_eff = xsecBeforeMatching_[i].value()>0? xsecAfterMatching_[i].value()/xsecBeforeMatching_[i].value(): 0;
	jetmatch_err = (xsecBeforeMatching_[i].value()>0 && xsecAfterMatching_[i].value()>0 && 
			pow(xsecAfterMatching_[i].error()/xsecAfterMatching_[i].value(),2)>pow(xsecBeforeMatching_[i].error()/xsecBeforeMatching_[i].value(),2)
			)? jetmatch_eff*sqrt(pow(xsecAfterMatching_[i].error()/xsecAfterMatching_[i].value(),2)-
					     pow(xsecBeforeMatching_[i].error()/xsecBeforeMatching_[i].value(),2)):-1;
      }
    else
      {
	title[i] = Form("%d",i);      
	jetmatch_eff = jetMatchEffStat_[i].filterEfficiency(hepidwtup_);
	jetmatch_err = jetMatchEffStat_[i].filterEfficiencyError(hepidwtup_);
      }


    edm::LogPrint("GenXSecAnalyzer") 
      << title[i] << "\t\t"
      << std::scientific << std::setprecision(3)
      << xsecBeforeMatching_[i].value()  << " +/- " 
      << xsecBeforeMatching_[i].error()  << "\t\t"
      << eventEffStat_[i].numEventsPassed() << "\t"
      << jetMatchEffStat_[i].numPassPositiveEvents() << "\t"
      << jetMatchEffStat_[i].numPassNegativeEvents() << "\t"
      << eventEffStat_[i].numEventsTotal() << "\t"
      << jetMatchEffStat_[i].numTotalPositiveEvents() << "\t"
      << jetMatchEffStat_[i].numTotalNegativeEvents() << "\t"
      << std::scientific << std::setprecision(3)
      << xsecAfterMatching_[i].value() << " +/- "
      << xsecAfterMatching_[i].error() << "\t\t"
      << std::fixed << std::setprecision(1)
      << (jetmatch_eff*100)  << " +/- " << (jetmatch_err*100) << "\t"
      << std::fixed << std::setprecision(1)
      << (eventEffStat_[i].filterEfficiency(+3) * 100) << " +/- " << ( eventEffStat_[i].filterEfficiencyError(+3) * 100);


  }
  delete [] title;

  edm::LogPrint("GenXSecAnalyzer") 
    << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------";

  edm::LogPrint("GenXSecAnalyzer") 
    << "Before matching: total cross section = " 
    << std::scientific << std::setprecision(3)  
    << xsecBeforeMatching_[last].value() << " +- " << xsecBeforeMatching_[last].error() <<  " pb";
  
  double xsec_match  = xsecAfterMatching_[last].value();
  double error_match = xsecAfterMatching_[last].error();

  edm::LogPrint("GenXSecAnalyzer") 
    << "After matching: total cross section = " 
    << std::scientific << std::setprecision(3)  
    << xsec_match << " +- " << error_match <<  " pb";


  // hepMC filter efficiency
  double hepMCFilter_eff = 1.0;
  double hepMCFilter_err = 0.0;
  if( hasHepMCFilterInfo_){
    hepMCFilter_eff = hepMCFilterEffStat_.filterEfficiency(hepidwtup_);
    hepMCFilter_err = hepMCFilterEffStat_.filterEfficiencyError(hepidwtup_);

    edm::LogPrint("GenXSecAnalyzer") 
      << "HepMC filter efficiency (taking into account weights)= "
      << "(" << hepMCFilterEffStat_.sumPassWeights() << ")"
      << " / "
      << "(" << hepMCFilterEffStat_.sumWeights() << ")"
      << " = " 
      <<  std::scientific << std::setprecision(3) 
      << hepMCFilter_eff << " +- " << hepMCFilter_err;

    double hepMCFilter_event_total = hepMCFilterEffStat_.numTotalPositiveEvents() + hepMCFilterEffStat_.numTotalNegativeEvents();
    double hepMCFilter_event_pass  = hepMCFilterEffStat_.numPassPositiveEvents() +  hepMCFilterEffStat_.numPassNegativeEvents();
    double hepMCFilter_event_eff   =  hepMCFilter_event_total > 0 ? hepMCFilter_event_pass/ hepMCFilter_event_total : 0;
    double hepMCFilter_event_err   = hepMCFilter_event_total > 0 ? 
      sqrt((1-hepMCFilter_event_eff)*hepMCFilter_event_eff/hepMCFilter_event_total): -1;
    edm::LogPrint("GenXSecAnalyzer") 
      << "HepMC filter efficiency (event-level)= " 
      << "(" << hepMCFilter_event_pass << ")"
      << " / "
      << "(" << hepMCFilter_event_total << ")"
      << " = " 
      <<  std::scientific << std::setprecision(3) 
      << hepMCFilter_event_eff << " +- " <<  hepMCFilter_event_err;
  }

  // gen-particle filter efficiency
  double filterOnly_eff = filterOnlyEffStat_.filterEfficiency(hepidwtup_);
  double filterOnly_err = filterOnlyEffStat_.filterEfficiencyError(hepidwtup_);

  edm::LogPrint("GenXSecAnalyzer") 
    << "Filter efficiency (taking into account weights)= "
    << "(" << filterOnlyEffStat_.sumPassWeights() << ")"
    << " / "
    << "(" << filterOnlyEffStat_.sumWeights() << ")"
    << " = " 
    <<  std::scientific << std::setprecision(3) 
    << filterOnly_eff << " +- " << filterOnly_err;

  double filterOnly_event_total = filterOnlyEffStat_.numTotalPositiveEvents() + filterOnlyEffStat_.numTotalNegativeEvents();
  double filterOnly_event_pass  = filterOnlyEffStat_.numPassPositiveEvents() + filterOnlyEffStat_.numPassNegativeEvents();
  double filterOnly_event_eff   = filterOnly_event_total > 0 ? filterOnly_event_pass/filterOnly_event_total : 0;
  double filterOnly_event_err   = filterOnly_event_total > 0 ?  
    sqrt((1-filterOnly_event_eff)*filterOnly_event_eff/filterOnly_event_total): -1;
  edm::LogPrint("GenXSecAnalyzer") 
    << "Filter efficiency (event-level)= " 
    << "(" << filterOnly_event_pass << ")"
    << " / "
    << "(" << filterOnly_event_total << ")"
    << " = " 
    <<  std::scientific << std::setprecision(3) 
    << filterOnly_event_eff << " +- " << filterOnly_event_err;


  double xsec_final  = xsec_match*filterOnly_eff*hepMCFilter_eff;
  double error_final = xsec_final*sqrt(error_match*error_match/xsec_match/xsec_match+
				       filterOnly_err*filterOnly_err/filterOnly_eff/filterOnly_eff + 
				       hepMCFilter_err*hepMCFilter_err/hepMCFilter_eff/hepMCFilter_eff
				       );

  edm::LogPrint("GenXSecAnalyzer") 
    << "After filter: final cross section = " 
    << std::scientific << std::setprecision(3)  
    << xsec_final << " +- " << error_final << " pb";

  xsec_ = GenLumiInfoProduct::XSec(xsec_final,error_final);


}


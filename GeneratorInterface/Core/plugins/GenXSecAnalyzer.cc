#include "GeneratorInterface/Core/interface/GenXSecAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "TMath.h"
#include <iostream>
#include <iomanip>

GenXSecAnalyzer::GenXSecAnalyzer(const edm::ParameterSet& iConfig):
  nMCs_(0),
  hepidwtup_(-9999),
  totalWeightPre_(0),
  thisRunWeightPre_(0),
  totalWeight_(0),
  thisRunWeight_(0),
  xsecPreFilter_(-1,-1),
  xsec_(-1,-1),
  product_(GenLumiInfoProduct(-9999)),
  filterOnlyEffRun_(0,0,0,0,0.,0.,0.,0.),
  hepMCFilterEffRun_(0,0,0,0,0.,0.,0.,0.),
  filterOnlyEffStat_(0,0,0,0,0.,0.,0.,0.),
  hepMCFilterEffStat_(0,0,0,0,0.,0.,0.,0.)
{
  xsecBeforeMatching_.clear();
  xsecAfterMatching_.clear(); 
  jetMatchEffStat_.clear(); 
  previousLumiBlockLHEXSec_.clear();
  currentLumiBlockLHEXSec_.clear();

  genFilterInfoToken_ = consumes<GenFilterInfo,edm::InLumi>(edm::InputTag("genFilterEfficiencyProducer",""));
  hepMCFilterInfoToken_ = consumes<GenFilterInfo,edm::InLumi>(edm::InputTag("generator",""));
  genLumiInfoToken_ = consumes<GenLumiInfoProduct,edm::InLumi>(edm::InputTag("generator",""));
}

GenXSecAnalyzer::~GenXSecAnalyzer()
{
}

void
GenXSecAnalyzer::beginJob() {

  xsecBeforeMatching_.clear();
  xsecAfterMatching_.clear(); 
  jetMatchEffStat_.clear(); 
  previousLumiBlockLHEXSec_.clear();
  currentLumiBlockLHEXSec_.clear();

}

void 
GenXSecAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const&)
{
  // initialization for every different physics MC

  nMCs_++;

  thisRunWeightPre_ = 0;
  thisRunWeight_ = 0;


  product_ = GenLumiInfoProduct(-9999);

  filterOnlyEffRun_ = GenFilterInfo(0,0,0,0,0.,0.,0.,0.);
  hepMCFilterEffRun_ = GenFilterInfo(0,0,0,0,0.,0.,0.,0.);


  xsecBeforeMatching_.clear();
  xsecAfterMatching_.clear(); 
  jetMatchEffStat_.clear(); 
  previousLumiBlockLHEXSec_.clear();
  currentLumiBlockLHEXSec_.clear();


  return;
}

void
GenXSecAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {

}



void
GenXSecAnalyzer::analyze(const edm::Event&, const edm::EventSetup&)
{
}



void
GenXSecAnalyzer::endLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {


  edm::Handle<GenLumiInfoProduct> genLumiInfo;
  iLumi.getByToken(genLumiInfoToken_,genLumiInfo);
  if (!genLumiInfo.isValid()) return;
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
    }


  for(unsigned int ip=0; ip < theProcesses_size; ip++)
    {

      if(theProcesses[ip].lheXSec().value()<0){
	edm::LogError("GenXSecAnalyzer::endLuminosityBlock") << "cross section of process " << ip << " value = "  << theProcesses[ip].lheXSec().value();
	return;
      }
    }

  product_.mergeProduct(*genLumiInfo);

  edm::Handle<GenFilterInfo> genFilter;
  iLumi.getByToken(genFilterInfoToken_,genFilter);
  if(genFilter.isValid())
    {
      filterOnlyEffStat_.mergeProduct(*genFilter);
      filterOnlyEffRun_.mergeProduct(*genFilter);
      thisRunWeight_ += genFilter->sumPassWeights();
    }


  edm::Handle<GenFilterInfo> hepMCFilter;
  iLumi.getByToken(hepMCFilterInfoToken_,hepMCFilter);

  if(hepMCFilter.isValid())
    {
      hepMCFilterEffStat_.mergeProduct(*hepMCFilter);
      hepMCFilterEffRun_.mergeProduct(*hepMCFilter);
    }
  

  // doing generic summing for jet matching statistics
  // and computation of combined LHE information
  for(unsigned int ip=0; ip < theProcesses_size; ip++)
    {
      int id = theProcesses[ip].process();
      GenFilterInfo&x = jetMatchEffStat_[id];
      GenLumiInfoProduct::XSec& y = currentLumiBlockLHEXSec_[id];
      GenLumiInfoProduct::FinalStat temp_killed   = theProcesses[ip].killed();
      GenLumiInfoProduct::FinalStat temp_selected = theProcesses[ip].selected();
      double passw  = temp_killed.sum();
      double passw2 = temp_killed.sum2();
      double totalw  = temp_selected.sum();
      double totalw2 = temp_selected.sum2();
      GenFilterInfo tempInfo(
			     theProcesses[ip].nPassPos(),
			     theProcesses[ip].nPassNeg(),
			     theProcesses[ip].nTotalPos(),
			     theProcesses[ip].nTotalNeg(),
			     passw,
			     passw2,
			     totalw,
			     totalw2);

    // matching statistics for all processes
      jetMatchEffStat_[10000].mergeProduct(tempInfo);
      double currentValue  = theProcesses[ip].lheXSec().value();
      double currentError  = theProcesses[ip].lheXSec().error();


      // this process ID has occurred before
      if(y.value()>0)
	{
	  x.mergeProduct(tempInfo);
	  double previousValue = previousLumiBlockLHEXSec_[id].value();

	  if(currentValue != previousValue) // transition of cross section
	    {

	      double xsec = y.value();
	      double err  = y.error();
	      combine(xsec, err, thisRunWeightPre_, currentValue, currentError, totalw);
	      y = GenLumiInfoProduct::XSec(xsec,err);
	    }
	  else // LHE cross section is the same as previous lumiblock
	    thisRunWeightPre_ += totalw;
	
	}
      // this process ID has never occurred before
      else
	{
	  x = tempInfo;
	  y = theProcesses[ip].lheXSec();	      
	  thisRunWeightPre_ += totalw;
	}

      previousLumiBlockLHEXSec_[id]= theProcesses[ip].lheXSec();
    } // end

 

  return;
  
}

void 
GenXSecAnalyzer::endRun(edm::Run const& iRun, edm::EventSetup const&)
{
  //xsection before matching
  edm::Handle<LHERunInfoProduct> run;

  if(iRun.getByLabel("externalLHEProducer", run ))
    {
      const lhef::HEPRUP thisHeprup_ = run->heprup();

      for ( unsigned int iSize = 0 ; iSize < thisHeprup_.XSECUP.size() ; iSize++ ) {
	std::cout  << std::setw(14) << std::fixed << thisHeprup_.XSECUP[iSize]
		   << std::setw(14) << std::fixed << thisHeprup_.XERRUP[iSize]
		   << std::setw(14) << std::fixed << thisHeprup_.XMAXUP[iSize]
		   << std::setw(14) << std::fixed << thisHeprup_.LPRUP[iSize] 
		   << std::endl;
      }
      std::cout << " " << std::endl;
    }
        

 
  // compute cross section for this run first
  // set the correct combined LHE+filter cross sections
  unsigned int i = 0;
  std::vector<GenLumiInfoProduct::ProcessInfo> newInfos;
  for(std::map<int, GenLumiInfoProduct::XSec>::const_iterator iter = currentLumiBlockLHEXSec_.begin();
      iter!=currentLumiBlockLHEXSec_.end(); ++iter, i++)
    {
      GenLumiInfoProduct::ProcessInfo temp = product_.getProcessInfos()[i];
      temp.setLheXSec(iter->second.value(),iter->second.error());
      newInfos.push_back(temp);
    }
  product_.setProcessInfo(newInfos);

  const GenLumiInfoProduct::XSec thisRunXSecPre     = compute(product_);
 // xsection after matching before filters
  combine(xsecPreFilter_, totalWeightPre_, thisRunXSecPre, thisRunWeightPre_);

  double thisHepFilterEff = 1; 
  double thisHepFilterErr = 0; 

  if(hepMCFilterEffRun_.sumWeights2()>0)
    {
      thisHepFilterEff = hepMCFilterEffRun_.filterEfficiency(hepidwtup_);
      thisHepFilterErr = hepMCFilterEffRun_.filterEfficiencyError(hepidwtup_);
      if(thisHepFilterEff<0)
	{
	  thisHepFilterEff = 1; 
	  thisHepFilterErr = 0; 
	}

    }

  double thisGenFilterEff = 1; 
  double thisGenFilterErr = 0; 

  if(filterOnlyEffRun_.sumWeights2()>0)
    {
      thisGenFilterEff = filterOnlyEffRun_.filterEfficiency(hepidwtup_);
      thisGenFilterErr = filterOnlyEffRun_.filterEfficiencyError(hepidwtup_);
      if(thisGenFilterEff<0)
	{
	  thisGenFilterEff = 1; 
	  thisGenFilterErr = 0; 
	}

    }
  double thisXsec = thisRunXSecPre.value() > 0 ? thisHepFilterEff*thisGenFilterEff*thisRunXSecPre.value() : 0;
  double thisErr  = thisRunXSecPre.value() > 0 ? thisXsec*
    sqrt(pow(TMath::Max(thisRunXSecPre.error(),(double)0)/thisRunXSecPre.value(),2)+
	 pow(thisHepFilterErr/thisHepFilterEff,2)+
	 pow(thisGenFilterErr/thisGenFilterEff,2)) : 0;
  const GenLumiInfoProduct::XSec thisRunXSec= GenLumiInfoProduct::XSec(thisXsec,thisErr);
  combine(xsec_, totalWeight_, thisRunXSec, thisRunWeight_);

}



void 
GenXSecAnalyzer::combine(double& finalValue, double& finalError, double& finalWeight, const double& currentValue, const double& currentError, const double & currentWeight)
{

  if(finalValue<1e-10)
    {
      finalValue = currentValue;
      finalError = currentError;
      finalWeight += currentWeight;
    }
  else
    {
      double wgt1 = (finalError < 1e-10 || currentError<1e-10)?
  	finalWeight :
  	1/(finalError*finalError);
      double wgt2 = (finalError < 1e-10 || currentError<1e-10)?
	currentWeight:
  	1/(currentError*currentError);
      double xsec = (wgt1 * finalValue + wgt2 * currentValue) /(wgt1 + wgt2);
      double err  = (finalError < 1e-10 || currentError<1e-10)? 0 : 
  	1.0 / std::sqrt(wgt1 + wgt2);
      finalValue = xsec;
      finalError = err;
      finalWeight += currentWeight;
    }
  return;
    
}

void 
GenXSecAnalyzer::combine(GenLumiInfoProduct::XSec& finalXSec, double &totalw, const GenLumiInfoProduct::XSec& thisRunXSec, const double& thisw)
{
  double value = finalXSec.value();
  double error = finalXSec.error();
  double thisValue = thisRunXSec.value();
  double thisError = thisRunXSec.error();
  combine(value,error,totalw,thisValue,thisError,thisw);
  finalXSec = GenLumiInfoProduct::XSec(value,error);
  return;
}


GenLumiInfoProduct::XSec 
GenXSecAnalyzer::compute(const GenLumiInfoProduct& iLumiInfo)
{
  // sum of cross sections and errors over different processes
  double sigSelSum = 0.0;
  double err2SelSum = 0.0;
  double sigSum = 0.0;
  double err2Sum = 0.0;

  std::vector<GenLumiInfoProduct::XSec> tempVector_before;
  std::vector<GenLumiInfoProduct::XSec> tempVector_after;

  // loop over different processes for each sample
  unsigned int vectorSize = iLumiInfo.getProcessInfos().size();
  for(unsigned int ip=0; ip < vectorSize; ip++){
    GenLumiInfoProduct::ProcessInfo proc = iLumiInfo.getProcessInfos()[ip];	  
    double hepxsec_value = proc.lheXSec().value();
    double hepxsec_error = proc.lheXSec().error() < 1e-10? 0:proc.lheXSec().error();
    tempVector_before.push_back(GenLumiInfoProduct::XSec(hepxsec_value,hepxsec_error));

    sigSelSum += hepxsec_value;
    err2SelSum += hepxsec_error*hepxsec_error;

    // skips computation if jet matching efficiency=0
    if (proc.killed().n()<1)
      {
	tempVector_after.push_back(GenLumiInfoProduct::XSec(0.0,0.0));
	continue;
      }

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
    
    if(fracAcc<1e-6)
      {
	tempVector_after.push_back(GenLumiInfoProduct::XSec(0.0,0.0));
	continue;
      }
    
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
			    
	efferr2 = denominator>0?
	  numerator/denominator:0;
	break;
      }
    }
    double delta2Veto = efferr2/fracAcc/fracAcc;
	
    // computing total error on cross section after matching efficiency
    
    double sigma2Sum, sigma2Err;
    sigma2Sum = hepxsec_value * hepxsec_value;
    sigma2Err = hepxsec_error * hepxsec_error;
 

    double delta2Sum = delta2Veto
      + sigma2Err / sigma2Sum;
    relErr = (delta2Sum > 0.0 ?
	      std::sqrt(delta2Sum) : 0.0);
    double deltaFin = sigmaFin * relErr;

    tempVector_after.push_back(GenLumiInfoProduct::XSec(sigmaFin,deltaFin));

    // sum of cross sections and errors over different processes
    sigSum += sigmaFin;
    err2Sum += deltaFin * deltaFin;

	

  } // end of loop over different processes
  tempVector_before.push_back(GenLumiInfoProduct::XSec(sigSelSum, sqrt(err2SelSum)));
  GenLumiInfoProduct::XSec result(sigSum,std::sqrt(err2Sum));
  tempVector_after.push_back(result);

  xsecBeforeMatching_ =tempVector_before;
  xsecAfterMatching_  =tempVector_after;

  return result;
}


void
GenXSecAnalyzer::endJob() {

  edm::LogPrint("GenXSecAnalyzer") << "\n"
				   << "------------------------------------" << "\n"
				   << "GenXsecAnalyzer:" << "\n"
				   << "------------------------------------";
 
  if(!jetMatchEffStat_.size()) {
    edm::LogPrint("GenXSecAnalyzer") << "------------------------------------" << "\n"
				     << "Cross-section summary not available" << "\n"
				     << "------------------------------------";
    return;
  }

 
  // below print out is only for combination of same physics MC samples and ME+Pythia MCs
 
  if(nMCs_==1 && hepidwtup_!=-1){

    edm::LogPrint("GenXSecAnalyzer") 
      << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- \n"
      << "Overall cross-section summary \n"
      << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------";
    edm::LogPrint("GenXSecAnalyzer")  
      << "Process\t\txsec_before [pb]\t\tpassed\tnposw\tnnegw\ttried\tnposw\tnnegw \txsec_match [pb]\t\t\taccepted [%]\t event_eff [%]";

    const unsigned sizeOfInfos = jetMatchEffStat_.size();
    const unsigned last = sizeOfInfos-1;
    std::string * title = new std::string[sizeOfInfos];
    unsigned int i = 0;
    double jetmatch_eff=0;
    double jetmatch_err=0;

    for(std::map<int, GenFilterInfo>::const_iterator iter = jetMatchEffStat_.begin();
  	iter!=jetMatchEffStat_.end(); ++iter, i++){ 

      GenFilterInfo thisJetMatchStat = iter->second;
      GenFilterInfo thisEventEffStat = GenFilterInfo(
						     thisJetMatchStat.numPassPositiveEvents()+thisJetMatchStat.numPassNegativeEvents(),
						     0,
						     thisJetMatchStat.numTotalPositiveEvents()+thisJetMatchStat.numTotalNegativeEvents(),
						     0,
						     thisJetMatchStat.numPassPositiveEvents()+thisJetMatchStat.numPassNegativeEvents(),
						     thisJetMatchStat.numPassPositiveEvents()+thisJetMatchStat.numPassNegativeEvents(),
						     thisJetMatchStat.numTotalPositiveEvents()+thisJetMatchStat.numTotalNegativeEvents(),
						     thisJetMatchStat.numTotalPositiveEvents()+thisJetMatchStat.numTotalNegativeEvents()
						     );


      if(i==last)
  	{
  	  title[i] = "Total";

  	  edm::LogPrint("GenXSecAnalyzer") 
  	    << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ";
 	
  	  double n1 = xsecBeforeMatching_[i].value();
  	  double e1 = xsecBeforeMatching_[i].error();
  	  double n2 = xsecAfterMatching_[i].value();
  	  double e2 = xsecAfterMatching_[i].error();

  	  jetmatch_eff = n1>0? n2/n1 : 0;
  	  jetmatch_err = (n1>0 && n2>0 && pow(e2/n2,2)>pow(e1/n1,2))?
  	    jetmatch_eff*sqrt( pow(e2/n2,2) - pow(e1/n1,2)):-1;
 	  
  	}
      else
  	{
  	  title[i] = Form("%d",i);      
  	  jetmatch_eff = thisJetMatchStat.filterEfficiency(hepidwtup_);
  	  jetmatch_err = thisJetMatchStat.filterEfficiencyError(hepidwtup_);

  	}

 
      edm::LogPrint("GenXSecAnalyzer") 
  	<< title[i] << "\t\t"
  	<< std::scientific << std::setprecision(3)
  	<< xsecBeforeMatching_[i].value()  << " +/- " 
  	<< xsecBeforeMatching_[i].error()  << "\t\t"
  	<< thisEventEffStat.numEventsPassed() << "\t"
  	<< thisJetMatchStat.numPassPositiveEvents() << "\t"
  	<< thisJetMatchStat.numPassNegativeEvents() << "\t"
  	<< thisEventEffStat.numEventsTotal() << "\t"
  	<< thisJetMatchStat.numTotalPositiveEvents() << "\t"
  	<< thisJetMatchStat.numTotalNegativeEvents() << "\t"
  	<< std::scientific << std::setprecision(3)
  	<< xsecAfterMatching_[i].value() << " +/- "
  	<< xsecAfterMatching_[i].error() << "\t\t"
  	<< std::fixed << std::setprecision(1)
  	<< (jetmatch_eff*100)  << " +/- " << (jetmatch_err*100) << "\t"
  	<< std::fixed << std::setprecision(1)
  	<< (thisEventEffStat.filterEfficiency(+3) * 100) << " +/- " 
  	<< ( thisEventEffStat.filterEfficiencyError(+3) * 100);

    }
    delete [] title;

    edm::LogPrint("GenXSecAnalyzer") 
      << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------";

    edm::LogPrint("GenXSecAnalyzer") 
      << "Before matching: total cross section = " 
      << std::scientific << std::setprecision(3)  
      << xsecBeforeMatching_[last].value() << " +- " << xsecBeforeMatching_[last].error() <<  " pb";

    edm::LogPrint("GenXSecAnalyzer") 
      << "After matching: total cross section = " 
      << std::scientific << std::setprecision(3)  
      << xsecAfterMatching_[last].value() << " +- " << xsecAfterMatching_[last].error() <<  " pb";
  }
  else if(hepidwtup_ == -1 )
    edm::LogPrint("GenXSecAnalyzer") 
      << "Before Filtrer: total cross section = " 
      << std::scientific << std::setprecision(3)  
      << xsecPreFilter_.value() << " +- " << xsecPreFilter_.error() <<  " pb";

  // hepMC filter efficiency
  double hepMCFilter_eff = 1.0;
  double hepMCFilter_err = 0.0;
  if(hepMCFilterEffStat_.sumWeights2()>0){
    hepMCFilter_eff = hepMCFilterEffStat_.filterEfficiency(-1);
    hepMCFilter_err = hepMCFilterEffStat_.filterEfficiencyError(-1);
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
  if(filterOnlyEffStat_.sumWeights2()>0){
    double filterOnly_eff = filterOnlyEffStat_.filterEfficiency(-1);
    double filterOnly_err = filterOnlyEffStat_.filterEfficiencyError(-1);

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

  }

  edm::LogPrint("GenXSecAnalyzer") 
    << "After filter: final cross section = " 
    << std::scientific << std::setprecision(3)  
    << xsec_.value() << " +- " << xsec_.error() << " pb";



}


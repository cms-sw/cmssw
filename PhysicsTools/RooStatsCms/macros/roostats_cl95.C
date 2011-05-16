static const char* desc =
"=====================================================================\n"
"|                                                                    \n"
"|\033[1m        roostats_cl95.C  version 1.05                 \033[0m\n"
"|                                                                    \n"
"| Standard c++ routine for 95% C.L. limit calculation                \n"
"| for cross section in a 'counting experiment'                       \n"
"| Fully backwards-compatible with the CL95 macro                     \n"
"|                                                                    \n"
"| also known as 'CL95 with RooStats'                                 \n"
"|                                                                    \n"
"|\033[1m Gena Kukartsev, Stefan Schmitz, Gregory Schott       \033[0m\n"
"|                                                                    \n"
"| July  2010: first version                                          \n"
"| March 2011: restructuring, interface change, expected limits       \n"
"| May 2011:   added expected limit median,                           \n"
"|             68%, 95% quantile bands and actual coverage            \n"
"|                                                                    \n"
"=====================================================================\n"
"                                                                     \n"
"Prerequisites:                                                       \n"
"                ROOT version 5.27/06 or higher                       \n"
"                                                                     \n"
"                                                                     \n"
"                                                                     \n"
"The code should be compiled in ROOT:                                 \n"
"                                                                     \n"
"root -l                                                              \n"
"                                                                     \n"
".L roostats_cl95.C+                                                  \n"
"                                                                     \n"
"Usage:                                                               \n"
" Double_t             limit = roostats_cl95(ilum, slum, eff, seff, bck, sbck, n, gauss = false, nuisanceModel, method, plotFileName, seed); \n"
" LimitResult expected_limit = roostats_clm(ilum, slum, eff, seff, bck, sbck, ntoys, nuisanceModel, method, seed); \n"
" Double_t     average_limit = roostats_cla(ilum, slum, eff, seff, bck, sbck, nuisanceModel, method, seed); \n"
"                                                                     \n"
"Inputs:                                                              \n"
"       ilum          - Nominal integrated luminosity (pb^-1)         \n"
"       slum          - Absolute error on the integrated luminosity   \n"
"       eff           - Nominal value of the efficiency times         \n"
"                       acceptance (in range 0 to 1)                  \n"
"       seff          - Absolute error on the efficiency times        \n"
"                       acceptance                                    \n"
"       bck           - Nominal value of the background estimate      \n"
"       sbck          - Absolute error on the background              \n"
"       n             - Number of observed events (not used for the   \n"
"                       expected limit)                               \n"
"       ntoys         - Number of pseudoexperiments to perform for    \n"
"                       expected limit calculation)                   \n"
"       gauss         - if true, use Gaussian statistics for signal   \n"
"                       instead of Poisson; automatically false       \n"
"                       for n = 0.                                    \n"
"                       Always false for expected limit calculations  \n"
"       nuisanceModel - distribution function used in integration over\n"
"                       nuisance parameters:                          \n"
"                       0 - Gaussian (default), 1 - lognormal,        \n"
"                       2 - gamma;                                    \n"
"                       (automatically 0 when gauss == true)          \n"
"       method        - method of statistical inference:              \n"
"                       \"bayesian\"  - Bayesian with numeric         \n"
"                                     integration (default),          \n"
"                       \"workspace\" - only create workspace and save\n"
"                                     to file, no interval calculation\n"
"       plotFileName  - file name for the control plot to be created  \n"
"                       file name extension will define the format,   \n"
"                       <plot_cl95.pdf> is the default value,         \n"
"                       specify empty string if you do not want       \n"
"                       the plot to be created (saves time)           \n"
"       seed          - seed for random number generation,            \n"
"                       specify 0 for unique irreproducible seed      \n"
"                                                                     \n"
"                                                                     \n"
"The statistics model in this routine: the routine addresses the task \n"
"of a Bayesian evaluation of limits for a one-bin counting experiment \n"
"with systematic uncertainties on luminosity and efficiency for the   \n"
"signal and a global uncertainty on the expected background (implying \n"
"no correlated error on the luminosity for signal and  background,    \n"
"which will not be suitable for all use cases!). The observable is the\n"
"measured number of events.                                           \n"
"                                                                     \n"
"For more details see                                                 \n"
"        https://twiki.cern.ch/twiki/bin/view/CMS/RooStatsCl95        \n"
"                                                                     \n"
"\033[1m       Note!                                           \033[0m\n"
"If you are running nonstandard ROOT environment, e.g. in CMSSW,      \n"
"you need to make sure that the RooFit and RooStats header files      \n"
"can be found since they might be in a nonstandard location.          \n"
"                                                                     \n"
"For CMSSW_4_2_0_pre8 and later, add the following line to your       \n"
"rootlogon.C:                                                         \n"
"      gSystem -> SetIncludePath( \"-I$ROOFITSYS/include\" );         \n";


#include <algorithm>

#include "TCanvas.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TUnixSystem.h"
#include "TStopwatch.h"

#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooDataSet.h"
#include "RooFitResult.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/SimpleInterval.h"
#include "RooStats/BayesianCalculator.h"
#include "RooStats/MCMCCalculator.h"
#include "RooStats/MCMCInterval.h"
#include "RooStats/MCMCIntervalPlot.h"
#include "RooStats/ProposalHelper.h"
#include "RooRandom.h"

// FIXME: remove namespaces
using namespace RooFit;
using namespace RooStats;
using namespace std;

class LimitResult;

Double_t roostats_cl95(Double_t ilum, Double_t slum,
		       Double_t eff, Double_t seff,
		       Double_t bck, Double_t sbck,
		       Int_t n,
		       Bool_t gauss = kFALSE,
		       Int_t nuisanceModel = 0,
		       std::string method = "bayesian",
		       std::string plotFileName = "plot_cl95.pdf",
		       UInt_t seed = 12345);

LimitResult roostats_clm(Double_t ilum, Double_t slum,
			 Double_t eff, Double_t seff,
			 Double_t bck, Double_t sbck,
			 Int_t nit = 200, Int_t nuisanceModel = 0,
			 std::string method = "bayesian",
			 UInt_t seed = 12345);

// legacy support: use roostats_clm() instead
Double_t roostats_cla(Double_t ilum, Double_t slum,
		      Double_t eff, Double_t seff,
		      Double_t bck, Double_t sbck,
		      Int_t nuisanceModel = 0,
		      std::string method = "bayesian",
		      UInt_t seed = 12345);




// ---> implementation below --------------------------------------------


class LimitResult{

  friend class CL95Calc;
  
public:
  LimitResult():
    _expected_limit(0),
    _low68(0),
    _high68(0),
    _low95(0),
    _high95(0){};

  ~LimitResult(){};

  Double_t GetExpectedLimit(){return _expected_limit;};

  Double_t GetOneSigmaLowRange(){return _low68;};
  Double_t GetOneSigmaHighRange(){return _high68;};
  Double_t GetOneSigmaCoverage(){return _cover68;};

  Double_t GetTwoSigmaLowRange(){return _low95;};
  Double_t GetTwoSigmaHighRange(){return _high95;};
  Double_t GetTwoSigmaCoverage(){return _cover95;};

private:
  Double_t _expected_limit;
  Double_t _low68;
  Double_t _high68;
  Double_t _low95;
  Double_t _high95;
  Double_t _cover68;
  Double_t _cover95;
};


class CL95Calc{

public:
  CL95Calc();
  CL95Calc( UInt_t seed );
  ~CL95Calc();

  RooWorkspace * makeWorkspace(Double_t ilum, Double_t slum,
			       Double_t eff, Double_t seff,
			       Double_t bck, Double_t sbck,
			       Bool_t gauss,
			       Int_t nuisanceModel);
  RooWorkspace * getWorkspace(){ return ws;}

  RooAbsData * makeData(Int_t n);

  Double_t cl95(std::string method = "bayesian");

  Double_t cla( Double_t ilum, Double_t slum,
		Double_t eff, Double_t seff,
		Double_t bck, Double_t sbck,
		Int_t nuisanceModel,
		std::string method );
  
  LimitResult clm(Double_t ilum, Double_t slum,
		  Double_t eff, Double_t seff,
		  Double_t bck, Double_t sbck,
		  Int_t nit = 200, Int_t nuisanceModel = 0,
		  std::string method = "bayesian");
  
  int makePlot( std::string method,
		std::string plotFileName = "plot_cl95.pdf" );

private:

  void init( UInt_t seed ); //  to be called by constructor

  // methods
  Double_t GetRandom( std::string pdf, std::string var );
  Long64_t LowBoundarySearch(std::vector<Double_t> * cdf, Double_t value);
  Long64_t HighBoundarySearch(std::vector<Double_t> * cdf, Double_t value);
  MCMCInterval * GetMcmcInterval(double conf_level,
				 int n_iter,
				 int n_burn,
				 double left_side_tail_fraction,
				 int n_bins);
  void makeMcmcPosteriorPlot( std::string filename );
  double printMcmcUpperLimit( std::string filename = "" );

  // data members
  RooWorkspace * ws;
  RooStats::ModelConfig mc;
  RooAbsData * data;
  BayesianCalculator * bcalc;
  RooStats::SimpleInterval * sInt;
  double nsig_rel_err;
  double nbkg_rel_err;
  Int_t _nuisance_model;

  // for Bayesian MCMC calculation
  MCMCInterval * mcInt;

  // random numbers
  TRandom3 r;

  // expected limits
  Double_t _expected_limit;
  Double_t _low68;
  Double_t _high68;
  Double_t _low95;
  Double_t _high95;
  
};


// default constructor
CL95Calc::CL95Calc(){
  init(0);
}


CL95Calc::CL95Calc(UInt_t seed){
  init(seed);
}


void CL95Calc::init(UInt_t seed){
  ws = new RooWorkspace("ws");
  data = 0;

  sInt = 0;
  bcalc = 0;
  mcInt = 0;
  mc.SetName("modelconfig");
  mc.SetTitle("ModelConfig for roostats_cl95");

  nsig_rel_err = -1.0; // default non-initialized value
  nbkg_rel_err = -1.0; // default non-initialized value

  // set random seed
  if (seed == 0){
    r.SetSeed();
    UInt_t _seed = r.GetSeed();
    UInt_t _pid = gSystem->GetPid();
    std::cout << "[CL95Calc]: random seed: " << _seed << std::endl;
    std::cout << "[CL95Calc]: process ID: " << _pid << std::endl;
    _seed = 31*_seed+_pid;
    std::cout << "[CL95Calc]: new random seed (31*seed+pid): " << _seed << std::endl;
    r.SetSeed(_seed);
    
    // set RooFit random seed (it has a private copy)
    RooRandom::randomGenerator()->SetSeed(_seed);
  }
  else{
    std::cout << "[CL95Calc]: random seed: " << seed << std::endl;
    r.SetSeed(seed);
    
    // set RooFit random seed (it has a private copy)
    RooRandom::randomGenerator()->SetSeed(seed);
  }

  // default Gaussian nuisance model
  _nuisance_model = 0;
}


CL95Calc::~CL95Calc(){
  delete ws;
  delete data;
  delete sInt;
  delete bcalc;
  delete mcInt;
}


RooWorkspace * CL95Calc::makeWorkspace(Double_t ilum, Double_t slum,
				       Double_t eff, Double_t seff,
				       Double_t bck, Double_t sbck,
				       Bool_t gauss,
				       Int_t nuisanceModel){

  if ( bck>0.0 && (sbck/bck)<5.0 ){
    // check that bck is not too close to zero,
    // so lognormal and gamma modls still make sense
    _nuisance_model = nuisanceModel;
  }
  else{
    _nuisance_model = 0;
    std::cout << "[CL95Calc]: background expectation is too close to zero compared to its uncertainty" << std::endl;
    std::cout << "[CL95Calc]: switching to the Gaussian nuisance model" << std::endl;

    // FIXME: is this appropriate fix for 0 bg expectation?
    if (bck<0.001){
      bck = std::max(bck,sbck/1000.0);
    }
  }

  // Workspace
  // RooWorkspace * ws = new RooWorkspace("ws",true);
  
  // observable: number of events
  ws->factory( "n[0]" );

  // integrated luminosity
  ws->factory( "lumi[0]" );

  // cross section - parameter of interest
  ws->factory( "xsec[0]" );

  // selection efficiency * acceptance
  ws->factory( "efficiency[0]" );

  // nuisance parameter: factor 1 with combined relative uncertainty
  ws->factory( "nsig_nuis[1.0]" ); // will adjust range below

  // signal yield
  ws->factory( "prod::nsig(lumi,xsec,efficiency, nsig_nuis)" );

  // estimated background yield
  ws->factory( "bkg_est[0]" );

  // nuisance parameter: factor 1 with background relative uncertainty
  //ws->factory( "nbkg_nuis[1.0]" ); // will adjust range below

  // background yield
  //ws->factory( "prod::nbkg(bkg_est, nbkg_nuis)" );
  ws->factory( "nbkg[1.0]" ); // will adjust value and range below

  // core model:
  ws->factory("sum::yield(nsig,nbkg)");
  if (gauss){
    // Poisson probability with mean signal+bkg
    ws->factory( "Gaussian::model_core(n,yield,expr('sqrt(yield)',yield))" );
  }
  else{
    // Poisson probability with mean signal+bkg
    ws->factory( "Poisson::model_core(n,yield)" );
  }


  // systematic uncertainties
  nsig_rel_err = sqrt(slum*slum/ilum/ilum+seff*seff/eff/eff);
  nbkg_rel_err = sbck/bck;
  if (_nuisance_model == 0){ // gaussian model for nuisance parameters

    std::cout << "[roostats_cl95]: Gaussian PDFs for nuisance parameters" << endl;

    // cumulative signal uncertainty
    ws->factory( "nsig_sigma[0.1]" );
    ws->factory( "Gaussian::syst_nsig(nsig_nuis, 1.0, nsig_sigma)" );
    // background uncertainty
    ws->factory( "nbkg_sigma[0.1]" );
    ws->factory( "Gaussian::syst_nbkg(nbkg, bkg_est, nbkg_sigma)" );

    ws->var("nsig_sigma")->setVal(nsig_rel_err);
    ws->var("nbkg_sigma")->setVal(sbck);
    ws->var("nsig_sigma")->setConstant(kTRUE);
    ws->var("nbkg_sigma")->setConstant(kTRUE);
  }
  else if (_nuisance_model == 1){// Lognormal model for nuisance parameters

    std::cout << "[roostats_cl95]: Lognormal PDFs for nuisance parameters" << endl;

    // cumulative signal uncertainty
    ws->factory( "nsig_kappa[1.1]" );
    ws->factory( "Lognormal::syst_nsig(nsig_nuis, 1.0, nsig_kappa)" );
    // background uncertainty
    ws->factory( "nbkg_kappa[1.1]" );
    ws->factory( "Lognormal::syst_nbkg(nbkg, bkg_est, nbkg_kappa)" );

    ws->var("nsig_kappa")->setVal(1.0 + nsig_rel_err);
    ws->var("nbkg_kappa")->setVal(1.0 + nbkg_rel_err);
    ws->var("nsig_kappa")->setConstant(kTRUE);
    ws->var("nbkg_kappa")->setConstant(kTRUE);
  }
  else if (_nuisance_model == 2){ // Gamma model for nuisance parameters

    std::cout << "[roostats_cl95]: Gamma PDFs for nuisance parameters" << endl;

    // cumulative signal uncertainty
    ws->factory( "nsig_beta[0.01]" );
    ws->factory( "nsig_gamma[101.0]" );
    ws->var("nsig_beta") ->setVal(nsig_rel_err*nsig_rel_err);
    ws->var("nsig_gamma")->setVal(1.0/nsig_rel_err/nsig_rel_err + 1.0);
    ws->factory( "Gamma::syst_nsig(nsig_nuis, nsig_gamma, nsig_beta, 0.0)" );

    // background uncertainty
    ws->factory( "nbkg_beta[0.01]" );
    ws->factory( "nbkg_gamma[101.0]" );
    ws->var("nbkg_beta") ->setVal(sbck*sbck/bck);
    ws->var("nbkg_gamma")->setVal(1.0/nbkg_rel_err/nbkg_rel_err + 1.0);
    ws->factory( "Gamma::syst_nbkg(nbkg, nbkg_gamma, nbkg_beta, 0.0)" );

    ws->var("nsig_beta") ->setConstant(kTRUE);
    ws->var("nsig_gamma")->setConstant(kTRUE);
    ws->var("nbkg_beta") ->setConstant(kTRUE);
    ws->var("nbkg_gamma")->setConstant(kTRUE);
  }
  else{
    std::cout <<"[roostats_cl95]: undefined nuisance parameter model specified, exiting" << std::endl;
  }

  // model with systematics
  ws->factory( "PROD::model(model_core, syst_nsig, syst_nbkg)" );

  // flat prior for the parameter of interest
  ws->factory( "Uniform::prior(xsec)" );  


  // parameter values
  ws->var("lumi")      ->setVal(ilum);
  ws->var("efficiency")->setVal(eff);
  ws->var("bkg_est")   ->setVal(bck);
  ws->var("xsec")      ->setVal(0.0);
  ws->var("nsig_nuis") ->setVal(1.0);
  ws->var("nbkg")      ->setVal(bck);

  // set some parameters as constants
  ws->var("lumi")      ->setConstant(kTRUE);
  ws->var("efficiency")->setConstant(kTRUE);
  ws->var("bkg_est")   ->setConstant(kTRUE);
  ws->var("n")         ->setConstant(kFALSE); // observable
  ws->var("xsec")      ->setConstant(kFALSE); // parameter of interest
  ws->var("nsig_nuis") ->setConstant(kFALSE); // nuisance
  ws->var("nbkg")      ->setConstant(kFALSE); // nuisance

  // floating parameters ranges
  // crude estimates! Need to know data to do better
  ws->var("n")        ->setRange( 0.0, bck+(5.0*sbck)+10.0); // ad-hoc range for obs
  ws->var("xsec")     ->setRange( 0.0, 15.0*(1.0+nsig_rel_err)/ilum/eff ); // ad-hoc range for POI
  ws->var("nsig_nuis")->setRange( std::max(0.0, 1.0 - 5.0*nsig_rel_err), 1.0 + 5.0*nsig_rel_err);
  ws->var("nbkg")     ->setRange( std::max(0.0, bck - 5.0*sbck), bck + 5.0*sbck);
  
  // Definition of observables and parameters of interest
  ws->defineSet("obsSet","n");
  ws->defineSet("poiSet","xsec");
  ws->defineSet("nuisanceSet","nsig_nuis,nbkg");

  // setup the ModelConfig object
  mc.SetWorkspace(*ws);
  mc.SetPdf(*(ws->pdf("model")));
  mc.SetParametersOfInterest(*(ws->set("poiSet")));
  mc.SetPriorPdf(*(ws->pdf("prior")));
  mc.SetNuisanceParameters(*(ws->set("nuisanceSet")));
  mc.SetObservables(*(ws->set("obsSet")));

  ws->import(mc);

  return ws;
}


RooAbsData * CL95Calc::makeData( Int_t n ){
  //
  // make the dataset owned by the class
  // the current one is deleted
  //
  // set ranges as well
  //
  
  // floating parameters ranges
  if (nsig_rel_err < 0.0 || nbkg_rel_err < 0.0){
    std::cout << "[roostats_cl95]: Workspace not initialized, cannot create a dataset" << std::endl;
    return 0;
  }
  
  double ilum = ws->var("lumi")->getVal();
  double eff  = ws->var("efficiency")->getVal();
  double bck  = ws->var("bkg_est")->getVal();
  double sbck = nbkg_rel_err*bck;

  ws->var("n")        ->setRange( 0.0, bck+(5.0*sbck)+10.0*(n+1.0)); // ad-hoc range for obs
  ws->var("xsec")     ->setRange( 0.0, 5.0*(1.0+nsig_rel_err)*std::max(10.0,n-bck)/ilum/eff ); // ad-hoc range for POI
  ws->var("nsig_nuis")->setRange( std::max(0.0, 1.0 - 5.0*nsig_rel_err), 1.0 + 5.0*nsig_rel_err);
  ws->var("nbkg")     ->setRange( std::max(0.0, bck - 5.0*sbck), bck + 5.0*sbck);

  // create data
  ws->var("n")         ->setVal(n);
  delete data;
  data = new RooDataSet("data","",*(mc.GetObservables()));
  data->add( *(mc.GetObservables()));

  return data;
}


MCMCInterval * CL95Calc::GetMcmcInterval(double conf_level,
					int n_iter,
					int n_burn,
					double left_side_tail_fraction,
					int n_bins){
  // use MCMCCalculator  (takes about 1 min)
  // Want an efficient proposal function, so derive it from covariance
  // matrix of fit
  
  RooFitResult * fit = ws->pdf("model")->fitTo(*data,Save());
  ProposalHelper ph;
  ph.SetVariables((RooArgSet&)fit->floatParsFinal());
  ph.SetCovMatrix(fit->covarianceMatrix());
  ph.SetUpdateProposalParameters(kTRUE); // auto-create mean vars and add mappings
  ph.SetCacheSize(100);
  ProposalFunction* pf = ph.GetProposalFunction();
  
  MCMCCalculator mcmc( *data, mc );
  mcmc.SetConfidenceLevel(conf_level);
  mcmc.SetNumIters(n_iter);          // Metropolis-Hastings algorithm iterations
  mcmc.SetProposalFunction(*pf);
  mcmc.SetNumBurnInSteps(n_burn); // first N steps to be ignored as burn-in
  mcmc.SetLeftSideTailFraction(left_side_tail_fraction);
  mcmc.SetNumBins(n_bins);
  
  delete mcInt;
  mcInt = mcmc.GetInterval();

  //std::cout << "!!!!!!!!!!!!!! interval" << std::endl;
  //if (mcInt == 0) std::cout << "!!!!!!!!!!!!!! no interval" << std::endl;
  
  return mcInt;
}


void CL95Calc::makeMcmcPosteriorPlot( std::string filename ){
  
  TCanvas c1("c1");
  MCMCIntervalPlot plot(*mcInt);
  plot.Draw();
  c1.SaveAs(filename.c_str());
  
  return;
}


double CL95Calc::printMcmcUpperLimit( std::string filename ){
  //
  // print out the upper limit on the first Parameter of Interest
  //

  RooRealVar * firstPOI = (RooRealVar*) mc.GetParametersOfInterest()->first();
  double _limit = mcInt->UpperLimit(*firstPOI);
  cout << "\n95% upper limit on " <<firstPOI->GetName()<<" is : "<<
    _limit <<endl;

  if (filename.size()!=0){
    
    std::ofstream aFile;

    // append to file if exists
    aFile.open(filename.c_str(), std::ios_base::app);

    char buf[1024];
    sprintf(buf, "%7.6f", _limit);

    aFile << buf << std::endl;

    // close outfile here so it is safe even if subsequent iterations crash
    aFile.close();

  }

  return _limit;
}


Double_t CL95Calc::cl95( std::string method ){

  // this method assumes that the workspace,
  // data and model config are ready

  Double_t upper_limit = -1.0;

  // make RooFit quiet
  RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

  if (method.find("bayesian") != std::string::npos){

    //prepare Bayesian Calulator
    delete bcalc;
    bcalc = new BayesianCalculator(*data, mc);
    TString namestring = "mybc";
    bcalc->SetName(namestring);
    bcalc->SetConfidenceLevel(0.95);
    bcalc->SetLeftSideTailFraction(0.0);
    
    delete sInt;
    sInt = bcalc->GetInterval();
    upper_limit = sInt->UpperLimit();
    delete sInt;
    sInt = 0;
   
  }
  else if (method.find("mcmc") != std::string::npos){

    std::cout << "[roostats_cl95]: Bayesian MCMC calculation is still experimental in this context!!!" << std::endl;

    //prepare Bayesian Markov Chain MC Calulator
    mcInt = GetMcmcInterval(0.95, 50000, 100, 0.0, 40);
    upper_limit = printMcmcUpperLimit();
  }
  else{
    std::cout << "[roostats_cl95]: method " << method 
	      << " is not implemented, exiting" <<std::endl;
    return -1.0;
  }
  
  return upper_limit;
  
}


Double_t CL95Calc::cla( Double_t ilum, Double_t slum,
			Double_t eff, Double_t seff,
			Double_t bck, Double_t sbck,
			Int_t nuisanceModel,
			std::string method ){

  makeWorkspace( ilum, slum,
		 eff, seff,
		 bck, sbck,
		 kFALSE,
		 nuisanceModel );
  
  Double_t CL95A = 0, precision = 1.e-4;

  Int_t i;
  for (i = bck; i >= 0; i--)
    {
      makeData( i );
      //
      Double_t s95 = cl95( method );
      Double_t s95w =s95*TMath::Poisson( (Double_t)i, bck );
      CL95A += s95w;
      cout << "[roostats_cla]: n = " << i << "; 95% C.L. = " << s95 << " pb; weighted 95% C.L. = " << s95w << " pb; running <s95> = " << CL95A << " pb" << endl;
      //
      if (s95w < CL95A*precision) break;
    }
  cout << "[roostats_cla]: Lower bound on n has been found at " << i+1 << endl;
  //
  for (i = bck+1; ; i++)
    {
      makeData( i );
      Double_t s95 = cl95( method );
      Double_t s95w =s95*TMath::Poisson( (Double_t)i, bck );
      CL95A += s95w;
      cout << "[roostats_cla]: n = " << i << "; 95% C.L. = " << s95 << " pb; weighted 95% C.L. = " << s95w << " pb; running <s95> = " << CL95A << " pb" << endl;
      //
      if (s95w < CL95A*precision) break;
    }
  cout << "[roostats_cla]: Upper bound on n has been found at " << i << endl;
  //
  cout << "[roostats_cla]: Average upper 95% C.L. limit = " << CL95A << " pb" << endl;

  return CL95A;
}



LimitResult CL95Calc::clm( Double_t ilum, Double_t slum,
			   Double_t eff, Double_t seff,
			   Double_t bck, Double_t sbck,
			   Int_t nit, Int_t nuisanceModel,
			   std::string method ){
  
  makeWorkspace( ilum, slum,
		 eff, seff,
		 bck, sbck,
		 kFALSE,
		 nuisanceModel );
  
  Double_t CLM = 0.0;
  LimitResult _result;

  Double_t b68[2] = {0.0, 0.0}; // 1-sigma expected band
  Double_t b95[2] = {0.0, 0.0}; // 2-sigma expected band

  std::vector<Double_t> pe;

  // timer
  TStopwatch t;
  t.Start(); // start timer
  Double_t _realtime = 0.0;
  Double_t _cputime = 0.0;
  Double_t _realtime_last = 0.0;
  Double_t _cputime_last = 0.0;
  Double_t _realtime_average = 0.0;
  Double_t _cputime_average = 0.0;

  // throw pseudoexperiments
  if (nit <= 0)return _result;
  for (Int_t i = 0; i < nit; i++)
    {
      // throw random nuisance parameter (bkg yield)
      Double_t bmean = GetRandom("syst_nbkg", "nbkg");

      std::cout << "[roostats_clm]: generatin pseudo-data with bmean = " << bmean << std::endl;
      Int_t n = r.Poisson(bmean);
      makeData( n );
      std::cout << "[roostats_clm]: invoking CL95 with n = " << n << std::endl;

      Double_t _pe = cl95( method );
      pe.push_back(_pe);
      CLM += pe[i];

      _realtime_last = t.RealTime() - _realtime;
      _cputime_last  = t.CpuTime() - _cputime;
      _realtime = t.RealTime();
      _cputime = t.CpuTime();
      t.Continue();
      _realtime_average = _realtime/((Double_t)(i+1));
      _cputime_average  = _cputime/((Double_t)(i+1));

      std::cout << "n = " << n << "; 95% C.L. = " << _pe << " pb; running <s95> = " << CLM/(i+1.) << std::endl;
      std::cout << "Real time (s), this iteration: " << _realtime_last << ", average per iteration: " << _realtime_average << ", total: " << _realtime << std::endl;
      std::cout << "CPU time (s),  this iteration: " << _cputime_last << ", average per iteration: " << _cputime_average << ", total: " << _cputime << std::endl << std::endl;
    }

  CLM /= nit;

  // sort the vector with limits
  std::sort(pe.begin(), pe.end());

  // median for the expected limit
  Double_t _median = TMath::Median(nit, &pe[0]);

  // quantiles for the expected limit bands
  Double_t _prob[4]; // array with quantile boundaries
  _prob[0] = 0.021;
  _prob[1] = 0.159;
  _prob[2] = 0.841;
  _prob[3] = 0.979;

  Double_t _quantiles[4]; // array for the results

  TMath::Quantiles(nit, 4, &pe[0], _quantiles, _prob); // evaluate quantiles

  b68[0] = _quantiles[1];
  b68[1] = _quantiles[2];
  b95[0] = _quantiles[0];
  b95[1] = _quantiles[3]; 

  // let's get actual coverages now

  Long64_t lc68 = LowBoundarySearch(&pe, _quantiles[1]);
  Long64_t uc68 = HighBoundarySearch(&pe, _quantiles[2]);
  Long64_t lc95 = LowBoundarySearch(&pe, _quantiles[0]);
  Long64_t uc95 = HighBoundarySearch(&pe, _quantiles[3]);

  Double_t _cover68 = (nit - lc68 - uc68)*100./nit;
  Double_t _cover95 = (nit - lc95 - uc95)*100./nit;

  std::cout << "[CL95Calc::clm()]: median limit: " << _median << std::endl;
  std::cout << "[CL95Calc::clm()]: 1 sigma band: [" << b68[0] << "," << b68[1] << 
    "]; actual coverage: " << _cover68 << 
    "%; lower/upper percentile: " << lc68*100./nit <<"/" << uc68*100./nit << std::endl;
  std::cout << "[CL95Calc::clm()]: 2 sigma band: [" << b95[0] << "," << b95[1] << 
    "]; actual coverage: " << _cover95 << 
    "%; lower/upper percentile: " << lc95*100./nit <<"/" << uc95*100./nit << std::endl;

  t.Print();

  _result._expected_limit = _median;
  _result._low68  = b68[0];
  _result._high68 = b68[1];
  _result._low95  = b95[0];
  _result._high95 = b95[1];
  _result._cover68 = _cover68;
  _result._cover95 = _cover95;

  return _result;
}



int CL95Calc::makePlot( std::string method,
			std::string plotFileName ){

  if (method.find("bayesian") != std::string::npos){

    std::cout << "[roostats_cl95]: making Bayesian posterior plot" << endl;
  
    TCanvas c1("posterior");
    bcalc->SetScanOfPosterior(100);
    RooPlot * plot = bcalc->GetPosteriorPlot();
    plot->Draw();
    c1.SaveAs(plotFileName.c_str());
  }
  else if (method.find("mcmc") != std::string::npos){

    std::cout << "[roostats_cl95]: making Bayesian MCMC posterior plot" << endl;

    makeMcmcPosteriorPlot(plotFileName);
  
  }
  else{
    std::cout << "[roostats_cl95]: method " << method 
	      << "is not implemented, exiting" <<std::endl;
    return -1;
  }

  return 0;
}



Double_t CL95Calc::GetRandom( std::string pdf, std::string var ){
  //
  // generates a random number using a pdf in the workspace
  //
  
  // generate a dataset with one entry
  RooDataSet * _ds = ws->pdf(pdf.c_str())->generate(*ws->var(var.c_str()), 1);

  Double_t _result = ((RooRealVar *)(_ds->get(0)->first()))->getVal();
  delete _ds;

  return _result;
}


Long64_t CL95Calc::LowBoundarySearch(std::vector<Double_t> * cdf, Double_t value){
  //
  // return number of elements which are < value with precision 1e-10
  //

  Long64_t result = 0;
  std::vector<Double_t>::const_iterator i = cdf->begin();
  while( (*i<value) && fabs(*i-value)>1.0e-10 && (i!=cdf->end()) ){
    ++i;
    ++result;
  }
  return result;
}


Long64_t CL95Calc::HighBoundarySearch(std::vector<Double_t> * cdf, Double_t value){
  //
  // return number of elements which are > value with precision 1e-10
  //

  Long64_t result = 0;
  std::vector<Double_t>::const_iterator i = cdf->end();
  while(1){ // (*i<value) && (i!=cdf->begin()) ){
    --i;
    if (*i>value && fabs(*i-value)>1.0e-10 ){
      ++result;
    }
    else break;
    if (i==cdf->begin()) break;
  }
  return result;
}


Int_t banner(){
  //#define __ROOFIT_NOBANNER // banner temporary off
#ifndef __EXOST_NOBANNER
  std::cout << desc << std::endl;
#endif
  return 0 ;
}

static Int_t dummy_ = banner() ;



Double_t roostats_cl95(Double_t ilum, Double_t slum,
		       Double_t eff, Double_t seff,
		       Double_t bck, Double_t sbck,
		       Int_t n,
		       Bool_t gauss,
		       Int_t nuisanceModel,
		       std::string method,
		       std::string plotFileName,
		       UInt_t seed){

  std::cout << "[roostats_cl95]: estimating 95% C.L. upper limit" << endl;
  if (method.find("bayesian") != std::string::npos){
    std::cout << "[roostats_cl95]: using Bayesian calculation via numeric integration" << endl;
  }
  else if (method.find("mcmc") != std::string::npos){
    std::cout << "[roostats_cl95]: using Bayesian calculation via numeric integration" << endl;
  }
  else if (method.find("workspace") != std::string::npos){
    std::cout << "[roostats_cl95]: no interval calculation, only create and save workspace" << endl;
  }
  else{
    std::cout << "[roostats_cl95]: method " << method 
	      << " is not implemented, exiting" <<std::endl;
    return -1.0;
  }

  // some input validation
  if (n < 0){
    std::cout << "Negative observed number of events specified, exiting" << std::endl;
    return -1.0;
  }

  if (n == 0) gauss = kFALSE;

  if (gauss){
    nuisanceModel = 0;
    std::cout << "[roostats_cl95]: Gaussian statistics used" << endl;
  }
  else{
    std::cout << "[roostats_cl95]: Poisson statistics used" << endl;
  }
    
  // limit calculation
  CL95Calc theCalc(seed);
  RooWorkspace * ws = theCalc.makeWorkspace( ilum, slum,
					     eff, seff,
					     bck, sbck,
					     gauss,
					     nuisanceModel );
  RooDataSet * data = (RooDataSet *)( theCalc.makeData( n )->Clone() );
  data->SetName("observed_data");
  ws->import(*data);

  //ws->Print();

  ws->SaveAs("ws.root");

  // if only workspace requested, exit here
  if ( method.find("workspace") != std::string::npos ) return 0.0;

  std::cout << "[roostats_cl95]: Range of allowed cross section values: [" 
	    << ws->var("xsec")->getMin() << ", " 
	    << ws->var("xsec")->getMax() << "]" << std::endl;
  Double_t limit = theCalc.cl95( method );
  std::cout << "[roostats_cl95]: 95% C.L. upper limit: " << limit << std::endl;

  // check if the plot is requested
  if (plotFileName.size() != 0){
    theCalc.makePlot(method, plotFileName);
  }

  return limit;
}


Double_t roostats_cla(Double_t ilum, Double_t slum,
		      Double_t eff, Double_t seff,
		      Double_t bck, Double_t sbck,
		      Int_t nuisanceModel,
		      std::string method,
		      UInt_t seed){

  Double_t limit = -1.0;

  std::cout << "[roostats_cla]: estimating average 95% C.L. upper limit" << endl;
  if (method.find("bayesian") != std::string::npos){
    std::cout << "[roostats_cla]: using Bayesian calculation via numeric integration" << endl;
  }
  else if (method.find("mcmc") != std::string::npos){
    std::cout << "[roostats_cla]: using Bayesian calculation via numeric integration" << endl;
  }
  else{
    std::cout << "[roostats_cla]: method " << method 
	      << " is not implemented, exiting" <<std::endl;
    return -1.0;
  }

  std::cout << "[roostats_cla]: Poisson statistics used" << endl;
    
  CL95Calc theCalc(seed);
  limit = theCalc.cla( ilum, slum,
		       eff, seff,
		       bck, sbck,
		       nuisanceModel,
		       method );

  //std::cout << "[roostats_cla]: average 95% C.L. upper limit: " << limit << std::endl;

  return limit;
}



LimitResult roostats_clm(Double_t ilum, Double_t slum,
			 Double_t eff, Double_t seff,
			 Double_t bck, Double_t sbck,
			 Int_t nit, Int_t nuisanceModel,
			 std::string method,
			 UInt_t seed){
  
  //Double_t limit = -1.0;
  LimitResult limit;

  std::cout << "[roostats_clm]: estimating average 95% C.L. upper limit" << endl;
  if (method.find("bayesian") != std::string::npos){
    std::cout << "[roostats_clm]: using Bayesian calculation via numeric integration" << endl;
  }
  else{
    std::cout << "[roostats_clm]: method " << method 
	      << "is not implemented, exiting" <<std::endl;
    //return -1.0;
    return limit;
  }

  std::cout << "[roostats_clm]: Poisson statistics used" << endl;
    
  CL95Calc theCalc(seed);
  limit = theCalc.clm( ilum, slum,
		       eff, seff,
		       bck, sbck,
		       nit, nuisanceModel,
		       method );

  return limit;
}

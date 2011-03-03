/////////////////////////////////////////////////////////////////////////
//
// first version of 'CL95 with RooStats' macro 
// authors: Stefan A. Schmitz, Gregory Schott
// date: July 2010
//
/////////////////////////////////////////////////////////////////////////

// running the macro:
// root -l
// .L roostats_cl95_bc.C
// roostats_cl95_bc( <your parameters> )
// e.g.: roostats_cl95_bc(10.0, 0.1, 0.51 , 0.15, 0.52, 0.3, 3);

// about the statistics model in this macro:
// This macro addresses the task of a Bayesian evaluation of limits for (one-bin) counting experiment with systematic uncertainties on luminosity and efficiency for the signal and a global uncertainty on the expected background (implying no correlated error o Luminosity for signal and  background, which will not be o.k. for all use cases!). The observable is the measured number of events. (Switching to alternative prior shapes is possible in RooStats. Changing the macro to use non-Gaussian priors is not difficult.) Here the bayesian 90% interval (upper limit corresponds to one-sided 95% upper limit) is evaluated as a function of the observed number of events in a hypothetical experiment.


#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooDataSet.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/SimpleInterval.h"
#include "RooStats/BayesianCalculator.h"

using namespace RooFit;
using namespace RooStats;
using namespace std;

void roostats_cl95_bc(double Lumi, double Lumi_err_rel, double eff, double eff_err_rel, double Nb, double Nb_err_rel, int obs){

   // Lumi : the integrated Luminosity in pb-1
   // Lumi_err_rel : relative error on Integrated Luminosity
   // eff : signal efficiency
   // eff_err_rel : relative error on signal efficiency
   // Nb : expected number of background events
   // Nb_err_rel : relative error on Nb
   // obs : observed number of events

   RooRealVar roo_Lumi("_Lumi","",Lumi);
   RooRealVar roo_Lumi_err_rel("_Lumi_err","",Lumi_err_rel*Lumi);
   RooRealVar roo_eff("_eff","",eff);
   RooRealVar roo_eff_err_rel("_eff_err","",eff_err_rel*eff);
   RooRealVar roo_Nb("_Nb","",Nb);
   RooRealVar roo_Nb_err_rel("_Nb_err","", Nb_err_rel*Nb);
   RooRealVar roo_obs("_obs","", obs);

   //define variables with ranges 


   RooRealVar roo_sigma_s("sigma_s","",0.0, 5.*((Nb+(5.*Nb_err_rel*Nb)+(obs+3.)+5.*sqrt(obs))/Lumi)+5. );
   //This does not work in all situations! In general one must check if the range is sufficiently large for receiving a stable result but not so large that one runs into technical problems with the integration. This would require an iterative adjustment  
   cout << "range for sigma_s (signal cross section): " <<  0.0 << " -- " <<  (5.*((Nb+(5.*Nb_err_rel*Nb)+(obs+3.)+5.*sqrt(obs))/Lumi)+5.) << endl;
   cout << "convince yourself that the range for the signal cross section is reasonable!" << endl;
   
   RooRealVar roo_n("n","",0,Nb+(5*Nb_err_rel*Nb)+10*(obs+1)); //this should ensure proper normalisation 

   RooRealVar roo_L("L","",TMath::Max(0.0,Lumi-(4*Lumi_err_rel*Lumi)),Lumi+(4*Lumi_err_rel*Lumi));
   RooRealVar roo_epsilon("epsilon","",TMath::Max(0.0,eff-(4*eff_err_rel*eff)),eff+(4*eff_err_rel*eff));
   RooRealVar roo_N_bkg("N_bkg","",TMath::Max(0.0,Nb-(4*Nb_err_rel*Nb)),Nb+(4*Nb_err_rel*Nb));

   // Definiton of a RooWorkspace containing the statistics model. 
   cout << "preparing the RooWorkspace object" << endl;

   RooWorkspace * myWS = new RooWorkspace("myWS",true);


   //load input parameters into workspace;
   myWS->import(roo_sigma_s);
   myWS->import(roo_L);
   myWS->import(roo_epsilon);
   myWS->import(roo_n);
   myWS->import(roo_N_bkg);

   //load input parameters into workspace;
   myWS->import(roo_Lumi);
   myWS->import(roo_Lumi_err_rel);
   myWS->import(roo_eff);
   myWS->import(roo_eff_err_rel);
   myWS->import(roo_Nb);
   myWS->import(roo_Nb_err_rel);
   myWS->import(roo_obs);

   // combined prior for signal contribution
   myWS->factory("Product::signal({sigma_s,L,epsilon})");
   //myWS->factory("N_bkg[0,3]");
   // define prior functions
   // uniform prior for signal crosssection
   myWS->factory("Uniform::prior_sigma_s(sigma_s)");
   // (truncated) prior for efficiency
   myWS->factory("Gaussian::prior_epsilon(epsilon,_eff,_eff_err)");
   // (truncated) Gaussian prior for luminosity
   myWS->factory("Gaussian::prior_L(L,_Lumi,_Lumi_err)");
   // (truncated) Gaussian prior for bkg crosssection
   myWS->factory("Gaussian::prior_N_bkg(N_bkg,_Nb,_Nb_err)");

   // Poisson distribution with mean signal+bkg
   myWS->factory("Poisson::model(n,sum(signal,N_bkg))");

   // define the global prior function 
   myWS->factory("PROD::prior(prior_sigma_s,prior_epsilon,prior_L,prior_N_bkg)");

   // Definition of observables and parameters of interest
   myWS->defineSet("obsSet","n");
   myWS->defineSet("poiSet","sigma_s");
   myWS->defineSet("nuisanceSet","N_bkg,L,epsilon");

   // ->model complete 

   // Currently the Bayesian methods will often not work well if the variable ranges 
   // are either too short (for obvious reasons) or too large (for technical reasons).
   // If the macro is not working well one should check if the variable ranges make 
   // sense for the specific application

   // A ModelConfig object is used to associate parts of your workspace with their statistical
   // meaning (it is also possible to initialize BayesianCalculator directly with elements from the
   // workspace but if you are sharing your workspace with others or if you want to use several
   // different methods the use of ModelConfig will most often turn out to be a good idea.)

   // setup the ModelConfig object
   cout << "preparing the ModelConfig object" << endl;

   ModelConfig modelconfig("modelconfig","ModelConfig for this example");
   modelconfig.SetWorkspace(*myWS);

   modelconfig.SetPdf(*(myWS->pdf("model")));
   modelconfig.SetParametersOfInterest(*(myWS->set("poiSet")));
   modelconfig.SetPriorPdf(*(myWS->pdf("prior")));
   modelconfig.SetNuisanceParameters(*(myWS->set("nuisanceSet")));
   modelconfig.SetObservables(*(myWS->set("obsSet")));


   // use BayesianCalculator to the derive confidence intervals as a function of the observed number of 

   cout << "starting the calculation of Bayesian confidence interval with BayesianCalculator" << endl;

   // prepare data input for the the observed number of events
   // adjust number of observed events in the workspace. This is communicated to ModelConfig!
   myWS->var("n")->setVal(obs);
   // create data
   RooDataSet data("data","",*(modelconfig.GetObservables()));
   data.add( *(modelconfig.GetObservables()));

   //prepare Bayesian Calulator
   BayesianCalculator bcalc(data, modelconfig);
   TString namestring = "mybc_";
   namestring += obs;
   bcalc.SetName(namestring);
   bcalc.SetConfidenceLevel(0.90);

   SimpleInterval* interval = bcalc.GetInterval();
   std::cout << "BayesianCalculator: 90% CL interval: [ " << interval->LowerLimit() << " - " << interval->UpperLimit() << " ] or 95% CL limits\n";


   delete interval; 
   


}

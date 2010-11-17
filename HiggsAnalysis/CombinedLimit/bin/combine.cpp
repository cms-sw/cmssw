/** 
   Generic limit setting, with any model.


   Takes as input:
    - one label for the output root file
    - one input datacard file
    - a value of the higgs mass
    - the name of the statistical method to apply
    - the number of toys to run (0 means to create a single toy corresponding to the expected signal, the so called Asimov dataset)
    - the seed for the random number generator

   
  The datacard can be:
    - a LandS-like datacard for a generic counting experiment with an arbitrary number of bins, processes and systematical uncertainties
    - a RooStats model in the "High Level Factory" syntax, that has to define:
      - a RooRealVar "r" corresponding to the cross section strength
      - a RooArgSet "obs" with the observables, and a RooArgSet "poi" with the parameters of interest (just "r")
      - RooAbsPdfs "model_s" and "model_b" for the (S+B) and B-only scenarios (not all statistical methods use both)
      - if systematical uncertainties are enabled, it must also define a RooArgSet "nuisances" with the nuisance parameters,
        and a RooAbsPdf "nuisancePdf" with the pdf for those. In this case "model_s" must already be the product of the pdf
        for the observables and the pdf for the nuisances.
      - the observed dataset will be constructed taking the default value of the observables as in the model.

  The program will assume that a file ending in ".hlf" is a RooStats model, and anything else is a LandS datacard.

  See higgsCombineSimple.cxx for the documentation of the other input parameters and of the output
*/
//#include "higgsCombine_Common.cxx"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <RooRandom.h>
#include <iostream>
#include <cstdlib>
#include <boost/program_options.hpp>
#include <string>

using namespace std;

int main(int argc, char **argv) {
  using namespace boost;
  namespace po = boost::program_options;

  string name;
  string datacard;
  int iMass;
  string whichMethod;
  unsigned int runToys;
  int    seed;
  bool   saveToys;
  string toysFile;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("name,n", po::value<string>(&name), "name")
    ("datacard,d", po::value<string>(&datacard), "datacard file")
    ("mass,m", po::value<int>(&iMass)->default_value(120), "minimum value for fit range")
    ("method,M", po::value<string>(&whichMethod)->default_value("mcmc"), "method to extract upper limit")
    ("toys,t", po::value<unsigned int>(&runToys)->default_value(0), "number of toy MC (0 = no toys)")
    ("seed,s", po::value<int>(&seed)->default_value(123456), "toy MC random seed")
    ("saveToys,w", po::value<bool>(&saveToys)->default_value(false), "save results of toy MC")
    ("toysFile,f", po::value<string>(&toysFile)->default_value(""), "toy MC output file")
    ;

  po::positional_options_description p;
  p.add("datacard", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).
	    options(desc).positional(p).run(), vm);
  po::notify(vm);
  
  if(vm.count("help")) {
    cout << "Usage: options_description [options]\n";
    cout << desc;
    return 0;
  }
  if(name == "") {
    cerr << "Missing name" << endl;
    cout << "Usage: options_description [options]\n";
    cout << desc;
    return 1001;
  }
  if(datacard == "") {
    cerr << "Missing datacard file" << endl;
    cout << "Usage: options_description [options]\n";
    cout << desc;
    return 1002;
  }

  bool doSyst = true;
  const string nosyst(".nosyst");
  size_t found = whichMethod.find(nosyst);
  if (found != string::npos) {
    whichMethod.replace(found, nosyst.length(),"");
    doSyst = false;
  }
  if      (whichMethod == "hybrid") method = hybrid;
  else if (whichMethod == "profileLikelihood") method = profileLikelihood;
  else if (whichMethod == "bayesianFlatPrior") method = bayesianFlatPrior;
  else if (whichMethod == "mcmc") method = mcmc;
  else if (whichMethod == "mcmcUniform") method = mcmcUniform;
  else {
    cerr << "Unsupported method: " << whichMethod << endl;
    cout << "Usage: options_description [options]\n";
    cout << desc;
    return 1003;
  }
  RooRandom::randomGenerator()->SetSeed(seed); 
  
  TString massName = TString::Format("mH%d.", iMass);
  TString toyName  = "";  if (runToys !=  0) toyName  = TString::Format("%d.", seed);
  TString fileName = "higgsCombine" + name + "."+whichMethod+"."+massName+toyName+"root";
  TFile *test = new TFile(fileName, "RECREATE");
  TTree *t = new TTree("test", "test");
  int syst, iToy, iChannel; 
  double mass, limit; 
  t->Branch("limit",&limit,"limit/D");
  t->Branch("mh",   &mass, "mh/D");
  t->Branch("syst", &syst, "syst/I");
  t->Branch("iToy", &iToy, "iToy/I");
  t->Branch("iChannel", &iChannel, "iChannel/I");
  t->Branch("t_cpu",   &t_cpu_,  "t_cpu/F");
  t->Branch("t_real",  &t_real_, "t_real/F");
  
  //if (saveToys) writeToysHere = new RooWorkspace("toys","toys"); 
  if (saveToys) writeToysHere = test->mkdir("toys","toys"); 
  if (toysFile != "") readToysFromHere = TFile::Open(TString(toysFile.c_str()));
  
  syst = doSyst;
  mass = iMass;
  iChannel = 0;
  combine(datacard, limit, iToy, t, runToys, syst);
  
  test->WriteTObject(t);
  test->Close();
}



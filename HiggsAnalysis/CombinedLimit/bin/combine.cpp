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
#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <RooRandom.h>
#include <iostream>
#include <cstdlib>

extern TString method;
extern Float_t t_cpu_, t_real_;
//RooWorkspace *writeToysHere = 0;
extern TDirectory *writeToysHere;
extern TDirectory *readToysFromHere;

void combine(TString hlfFile, double &limit, int &iToy, TTree *tree, int nToys=0, bool withSystematics=true);

void higgsCombineByHand(TString name, TString datacard, int iMass, TString whichMethod="mcmc", size_t runToys=0, int seed=42, bool saveToys=false, const char *toysFile = 0) {
    bool doSyst = true;
    method = whichMethod;
    if (method.Index(".nosyst") != -1) {
        method.ReplaceAll(".nosyst","");
        doSyst = false;
    }
    RooRandom::randomGenerator()->SetSeed(seed); 

    TString massName = TString::Format("mH%d.", iMass);
    TString toyName  = "";  if (runToys !=  0) toyName  = TString::Format("%d.",   seed);
    TString fileName = "higgsCombine"+name+"."+whichMethod+"."+massName+toyName+"root";
    TFile *test = new TFile(fileName,"RECREATE");
    TTree *t = new TTree("test","test");
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
    if (toysFile) readToysFromHere = TFile::Open(toysFile);

    syst = doSyst;
    mass = iMass;
    iChannel = 0;
    combine(datacard, limit, iToy, t, runToys, syst);

    test->WriteTObject(t);
    test->Close();
}
int main(int argc, char **argv) {
    if (argc < 4) { 
        std::cout << "higgsCombineByHand(TString name, TString datacard, int iMass, TString whichMethod=\"mcmc\", size_t runToys=0, int seed=42, bool saveToys=false)" << std::endl; 
        return 1; 
    }
    TString name(argv[1]);
    TString datacard(argv[2]);
    int iMass = atoi(argv[3]);
    TString whichMethod(argc > 4 ? argv[4] : "mcmc" );
    size_t runToys  =  (argc > 5 ? atoi(argv[5]) :  0);  
    int    seed     =  (argc > 6 ? atoi(argv[6]) : 42);
    bool   saveToys =  (argc > 7 ? atoi(argv[7]) :  0);
    const char *toysFile = (argc > 8 ? argv[8] : 0);
    higgsCombineByHand(name,datacard,iMass,whichMethod,runToys,seed,saveToys,toysFile);
}



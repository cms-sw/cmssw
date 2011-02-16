/**************************************
  Simple multiChannel significance & limit calculator
***************************************/
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include <cstring>
#include <cerrno>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

#include <TCanvas.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TIterator.h>
#include <TLine.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <TStopwatch.h>
#include <TTree.h>

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooMsgService.h>
#include <RooPlot.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooUniform.h>
#include <RooWorkspace.h>

#include <RooStats/HLFactory.h>
#include <RooStats/RooStatsUtils.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"

using namespace RooStats;
using namespace RooFit;

LimitAlgo * algo, * hintAlgo;

Float_t t_cpu_, t_real_;
TDirectory *outputFile = 0;
TDirectory *writeToysHere = 0;
TDirectory *readToysFromHere = 0;
int  verbose = 1;
bool withSystematics = 1;
bool doSignificance_ = 0;
float cl = 0.95;


Combine::Combine() :
    options_("Common options"), 
    rMin_(std::numeric_limits<float>::quiet_NaN()), 
    rMax_(std::numeric_limits<float>::quiet_NaN())
{
    namespace po = boost::program_options;

    options_.add_options()
        ("systematics,S", po::value<bool>(&withSystematics)->default_value(true), "Add systematic uncertainties")
        ("cl,C",   po::value<float>(&cl)->default_value(0.95), "Confidence Level")
        ("rMin",   po::value<float>(&rMin_), "Override minimum value for signal strength")
        ("rMax",   po::value<float>(&rMax_), "Override maximum value for signal strength")
        ("prior",  po::value<std::string>(&prior_)->default_value("flat"), "Prior to use, for methods that require it and if it's not already in the input file: 'flat' (default), '1/sqrt(r)'")
        ("compile", "Compile expressions instead of interpreting them")
        ("significance", "Compute significance instead of upper limit")
        ("hintStatOnly", "Ignore systematics when computing the hint")
        ("saveWorkspace", "Save workspace to output root file")
        ("toysNoSystematics", "Generate all toys with the central value of the nuisance parameters, without fluctuating them")
    ;
}

void Combine::applyOptions(const boost::program_options::variables_map &vm) 
{
  if(withSystematics) {
    std::cout << ">>> including systematics" << std::endl;
  } else {
    std::cout << ">>> no systematics included" << std::endl;
  } 
  compiledExpr_ = vm.count("compile");
  doSignificance_ = vm.count("significance");
  hintUsesStatOnly_ = vm.count("hintStatOnly");
  saveWorkspace_ = vm.count("saveWorkspace");
  toysNoSystematics_ = vm.count("toysNoSystematics");
}

bool Combine::mklimit(RooWorkspace *w, RooAbsData &data, double &limit) {
  TStopwatch timer;
  bool ret = false;
  try {
    double hint = 0; bool hashint = false;
    if (hintAlgo) {
        if (hintUsesStatOnly_ && withSystematics) {
            withSystematics = false;
            hashint = hintAlgo->run(w, data, hint, 0);
            withSystematics = true;
        } else {
            hashint = hintAlgo->run(w, data, hint, 0);
        } 
   }
    ret = algo->run(w, data, limit, (hashint ? &hint : 0));    
  } catch (std::exception &ex) {
    std::cerr << "Caught exception " << ex.what() << std::endl;
    return false;
  }
  if ((ret == false) && (verbose > 1)) {
    std::cout << "Failed for method " << algo->name() << "\n";
    std::cout << "  --- DATA ---\n";
    utils::printRAD(&data);
    std::cout << "  --- MODEL ---\n";
    w->Print("V");
  }
  timer.Stop(); t_cpu_ = timer.CpuTime()/60.; t_real_ = timer.RealTime()/60.;
  printf("Done in %.2f min (cpu), %.2f min (real)\n", t_cpu_, t_real_);
  return ret;
}

void Combine::run(TString hlfFile, const std::string &dataset, double &limit, int &iToy, TTree *tree, int nToys) {
  TString pwd(gSystem->pwd());
  TString tmpDir = "roostats-XXXXXX"; 
  mkdtemp(const_cast<char *>(tmpDir.Data()));
  gSystem->cd(tmpDir.Data());

  bool isTextDatacard = false, isBinary = false;
  TString fileToLoad = (hlfFile[0] == '/' ? hlfFile : pwd+"/"+hlfFile);
  if (!boost::filesystem::exists(fileToLoad.Data())) throw std::invalid_argument(("File "+fileToLoad+" does not exist").Data());
  if (hlfFile.EndsWith(".hlf") ) {
    // nothing to do
  } else if (hlfFile.EndsWith(".root")) {
    isBinary = true;
  } else {
    isTextDatacard = true;
    TString txtFile = fileToLoad.Data();
    TString options = "";
    if (!withSystematics) options += " --stat ";
    if (compiledExpr_)    options += " --compiled ";
    int status = gSystem->Exec("python -m HiggsAnalysis.CombinedLimit.lands2hlf "+options+" '"+txtFile+"' > model.hlf"); 
    fileToLoad = "model.hlf";
    if (status != 0 || !boost::filesystem::exists(fileToLoad.Data())) {
        throw std::invalid_argument("Failed to convert the input datacard from LandS to RooStats format. The lines above probably contain more information about the error.");
    }
  }

  if (getenv("CMSSW_BASE")) {
      if (verbose > 1) std::cout << "CMSSW_BASE is set, so will try to get include dir for roofit from scram." << std::endl;
      FILE *pipe = popen("scram tool tag roofitcore INCLUDE", "r"); 
      if (pipe) {
          char buff[1023];
          if (fgets(buff, 1023, pipe)) {
              if (buff[0] == '/') {
                  // must also remove the line break
                  int ilast = strlen(buff)-1;
                  while (ilast > 0 && isspace(buff[ilast])) { buff[ilast--] = '\0'; }
                  // then pass it to root
                  gSystem->AddIncludePath(TString::Format(" -I%s ", buff));
                  if (verbose > 1) std::cout << "Adding " << buff << " to include path" << std::endl;
              } else { std::cout << "scram tool tag roofitcore INCLUDE returned " << buff << " which doesn't look like an include dir." << std::endl; }
          } else { std::cerr << "Failed to read from pipe 'scram tool tag roofitcore INCLUDE'" << std::endl; }
          pclose(pipe);
      } else { std::cerr << "Failed to invoke 'scram tool tag roofitcore INCLUDE'" << std::endl; }
  }

  if (verbose <= 1) RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
  // Load the model, but going in a temporary directory to avoid polluting the current one with garbage from 'cexpr'
  RooWorkspace *w = 0;
  std::auto_ptr<RooStats::HLFactory> hlf(0);
  if (isBinary) {
    TFile *fIn = TFile::Open(fileToLoad);
    w = dynamic_cast<RooWorkspace *>(fIn->Get("w"));
    if (w == 0) {  std::cerr << "Could not find workspace 'w' in file " << fileToLoad << std::endl; fIn->ls(); return; }
    w = dynamic_cast<RooWorkspace *>(w->Clone());
  } else {
    hlf.reset(new RooStats::HLFactory("factory", fileToLoad));
    gSystem->cd(pwd);
    w = hlf->GetWs();
    if (w == 0) {
        std::cerr << "Could not read HLF from file " <<  (hlfFile[0] == '/' ? hlfFile : pwd+"/"+hlfFile) << std::endl;
        return;
    }
  }

  if (verbose <= 1) RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);

  const RooArgSet * observables = w->set("observables");
  if (observables == 0) throw std::invalid_argument("The model must define a RooArgSet 'observables'");

  if (w->pdf("model_s") == 0) throw std::invalid_argument("The model must define a RooAbsPdf 'model_s'");

  if (w->var("r") == 0 || w->set("POI") == 0) {
    throw std::invalid_argument("The model must define a RooRealVar 'r' for the signal strength, and a RooArgSet 'POI' with the parameters of interest.");
  }


  if (w->data(dataset.c_str()) == 0) {
    if (w->pdf("model_b")->canBeExtended()) {
      std::cout << "Dataset " << dataset.c_str() << " not found, will try to generate the expected dataset from the 'model_b' pdf" << std::endl;
      RooDataHist *bdata_obs = w->pdf("model_b")->generateBinned(*observables, RooFit::Asimov()); 
      bdata_obs->SetName(dataset.c_str());
      w->import(*bdata_obs);
    } else {
      if (!isTextDatacard) std::cout << "Dataset " << dataset.c_str() << " not found, will make one from the starting values of the observables in the model." << std::endl;
      RooDataSet *data_obs = new RooDataSet(dataset.c_str(), dataset.c_str(), *observables); 
      data_obs->add(*observables);
      w->import(*data_obs);
    }
  }

  if (verbose < -1) {
      RooMsgService::instance().setStreamStatus(0,kFALSE);
      RooMsgService::instance().setStreamStatus(1,kFALSE);
      RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
  }

  const RooArgSet * nuisances   = w->set("nuisances");

  if (!isnan(rMin_)) w->var("r")->setMin(rMin_);
  if (!isnan(rMax_)) w->var("r")->setMax(rMax_);

  if (prior_ == "flat") {
    w->factory("Uniform::prior(r)");
  } else if (prior_ == "1/sqrt(r)") {
    w->factory("EXPR::prior(\"1/sqrt(r)\",r)");
  } else if (!prior_.empty() && w->pdf(prior_.c_str()) != 0) {
    std::cout << "Will use prior '" << prior_ << "' in from the input workspace" << std::endl;
  } else {
    std::cerr << "Unknown prior '" << prior_ << "'. It's not 'flat' '1/sqrt(r)' or the name of a pdf in the model.\n" << std::endl;
    throw std::invalid_argument("Bad prior");
  }

  if (withSystematics && nuisances == 0) {
    RooArgSet * nuisancesGuess = w->pdf("model_s")->getParameters(*observables);
    RooStats::RemoveConstantParameters(nuisancesGuess);
    nuisancesGuess->remove(*w->set("POI"), true, true);
    nuisancesGuess->setName("nuisances");
    if (nuisancesGuess->getSize() > 0) {
        std::cout << "Guessing the nuisances from the parameters of the model after removing observables and POIs: " << std::endl;
        w->import(*nuisancesGuess);
        nuisancesGuess->Print();
        nuisances = w->set("nuisances");
    } else {
        throw std::logic_error("The signal model has no nuisance parameters. Please run the limit tool with no systematics (option -S 0).");
    }
    delete nuisancesGuess;
  }  
  if (!withSystematics && nuisances != 0) {
    std::cout << "Will set nuisance parameters to constants: " ;
    std::auto_ptr<TIterator> iter(nuisances->createIterator());
    for (TObject *a = iter->Next(); a != 0; a = iter->Next()) {
       RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
       if (rrv) { rrv->setConstant(true); std::cout << " " << rrv->GetName(); }
    }
    std::cout << std::endl;
  }

  w->saveSnapshot("clean", w->allVars());

  if (saveWorkspace_) {
    outputFile->WriteTObject(w,"w");
  }

  if (nToys <= 0) { // observed or asimov
    iToy = nToys;
    RooAbsData *dobs = w->data(dataset.c_str());
    if (iToy == -1) {	
        if (w->pdf("model_b")->canBeExtended()) {
	  dobs = w->pdf("model_b")->generateBinned(*observables,RooFit::Extended(),RooFit::Asimov());
	} else {
	  dobs = w->pdf("model_b")->generate(*observables,1,RooFit::Asimov());
	}
    } else if (dobs == 0) {
      std::cerr << "No observed data '" << dataset << "' in the workspace. Cannot compute limit.\n" << std::endl;
      return;
    }
    std::cout << "Computing limit starting from " << (iToy == 0 ? "observation" : "expected outcome") << std::endl;
    if (verbose > 0) utils::printRAD(dobs);
    if (mklimit(w,*dobs,limit)) tree->Fill();
  }
  
  
  std::vector<double> limitHistory;
  if (nToys > 0) {
    double expLimit = 0;
    unsigned int nLimits = 0;
    w->loadSnapshot("clean");
    RooDataSet *systDs = 0;
    if (withSystematics && !toysNoSystematics_ && (readToysFromHere == 0)) {
      if (nuisances == 0 || w->pdf("nuisancePdf") == 0) {
        throw std::logic_error("Running with systematics enabled, but nuisances or nuisancePdf not defined.");
      }
      systDs = w->pdf("nuisancePdf")->generate(*nuisances, nToys);
    }
    for (iToy = 1; iToy <= nToys; ++iToy) {
      RooAbsData *absdata_toy = 0;
      if (readToysFromHere == 0) {
	w->loadSnapshot("clean");
	if (verbose > 1) utils::printPdf(w, "model_b");
	if (withSystematics && !toysNoSystematics_) {
	  std::auto_ptr<RooArgSet> vars(w->pdf("model_b")->getVariables());
	  *vars = *systDs->get(iToy-1);
	  if (verbose > 0) utils::printPdf(w, "model_b");
	}
	std::cout << "Generate toy " << iToy << "/" << nToys << std::endl;
	if (w->pdf("model_b")->canBeExtended()) {
	  RooDataHist *bdata_toy = w->pdf("model_b")->generateBinned(*observables,RooFit::Extended());
	  absdata_toy = bdata_toy;
	} else {
	  RooDataSet *data_toy = w->pdf("model_b")->generate(*observables,1);
	  absdata_toy = data_toy;
	}
      } else {
	absdata_toy = dynamic_cast<RooAbsData *>(readToysFromHere->Get(TString::Format("toys/toy_%d",iToy)));
	if (absdata_toy == 0) {
	  std::cerr << "Toy toy_"<<iToy<<" not found in " << readToysFromHere->GetName() << ". List follows:\n";
	  readToysFromHere->ls();
	  return;
	}
      }
      if (verbose > 0) utils::printRAD(absdata_toy);
      w->loadSnapshot("clean");
      if (verbose > 1) utils::printPdf(w, "model_b");
      if (mklimit(w,*absdata_toy,limit)) {
	tree->Fill();
	++nLimits;
	expLimit += limit; 
        limitHistory.push_back(limit);
      }
      if (writeToysHere) {
	//writeToysHere->import(*absdata_toy, RooFit::Rename(TString::Format("toy_%d", iToy)), RooFit::Silence());
	writeToysHere->WriteTObject(absdata_toy, TString::Format("toy_%d", iToy));
      }
      delete absdata_toy;
    }
    expLimit /= nLimits;
    double rms = 0;
    for (std::vector<double>::const_iterator itl = limitHistory.begin(); itl != limitHistory.end(); ++itl) {
        rms += pow(*itl-expLimit, 2);
    }
    if (nLimits > 1) {
        rms = sqrt(rms/(nLimits-1)/nLimits);
        cout << "mean   expected limit: r < " << expLimit << " +/- " << rms << " @ " << cl*100 << "%CL (" <<nLimits << " toyMC)" << endl;
    } else {
        cout << "mean   expected limit: r < " << expLimit << " @ " << cl*100 << "%CL (" <<nLimits << " toyMC)" << endl;
    }
    sort(limitHistory.begin(), limitHistory.end());
    if (nLimits > 0) {
        double medianLimit = (nLimits % 2 == 0 ? 0.5*(limitHistory[nLimits/2]+limitHistory[nLimits/2+1]) : limitHistory[nLimits/2]);
        cout << "median expected limit: r < " << medianLimit << " @ " << cl*100 << "%CL (" <<nLimits << " toyMC)" << endl;
        double hi68 = limitHistory[min<int>(nLimits-1,  ceil(0.84  * nLimits))];
        double lo68 = limitHistory[min<int>(nLimits-1, floor(0.16  * nLimits))];
        double hi95 = limitHistory[min<int>(nLimits-1,  ceil(0.975 * nLimits))];
        double lo95 = limitHistory[min<int>(nLimits-1, floor(0.025 * nLimits))];
        cout << "   68% expected band : " << lo68 << " < r < " << hi68 << endl;
        cout << "   95% expected band : " << lo95 << " < r < " << hi95 << endl;
    }
  }

  boost::filesystem::remove_all(tmpDir.Data());
}


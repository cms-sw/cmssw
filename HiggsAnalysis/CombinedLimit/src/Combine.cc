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
#include <exception>
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
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"

using namespace RooStats;
using namespace RooFit;

LimitAlgo * algo, * hintAlgo;

Float_t t_cpu_, t_real_;
TDirectory *writeToysHere = 0;
TDirectory *sprnwriteToysHere = 0;
TDirectory *readToysFromHere = 0;
int  verbose = 1;
bool withSystematics = 1;
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
    ;
}

void Combine::applyOptions(const boost::program_options::variables_map &vm) 
{
  if(withSystematics) {
    std::cout << ">>> including systematics" << std::endl;
  } else {
    std::cout << ">>> no systematics included" << std::endl;
  }
}

bool Combine::mklimit(RooWorkspace *w, RooAbsData &data, double &limit) {
  TStopwatch timer;
  bool ret = false;
  try {
    double hint = 0; bool hashint = false;
    if (hintAlgo) {
        hashint = hintAlgo->run(w, data, hint, 0);
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

  TString fileToLoad;
  if (hlfFile.EndsWith(".hlf")) {
    fileToLoad = (hlfFile[0] == '/' ? hlfFile : pwd+"/"+hlfFile);
  }  else {
    TString txtFile = (hlfFile[0] == '/' ? hlfFile : pwd+"/"+hlfFile);
    if (!withSystematics) {
      gSystem->Exec("python -m HiggsAnalysis.CombinedLimit.lands2hlf --stat '"+txtFile+"' > model.hlf"); 
    } else {
      gSystem->Exec("python -m HiggsAnalysis.CombinedLimit.lands2hlf '"+txtFile+"' > model.hlf"); 
    }
    fileToLoad = "model.hlf";
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

  // Load the model, but going in a temporary directory to avoid polluting the current one with garbage from 'cexpr'
  RooStats::HLFactory hlf("factory", fileToLoad);
  gSystem->cd(pwd);
  RooWorkspace *w = hlf.GetWs();
  if (w == 0) {
    std::cerr << "Could not read HLF from file " <<  (hlfFile[0] == '/' ? hlfFile : pwd+"/"+hlfFile) << std::endl;
    return;
  }
  if (w->data(dataset.c_str()) == 0) {
    if (w->pdf("model_b")->canBeExtended()) {
      RooDataHist *bdata_obs = w->pdf("model_b")->generateBinned(*w->set("observables"), RooFit::Asimov()); 
      bdata_obs->SetName(dataset.c_str());
      w->import(*bdata_obs);
    } else {
      RooDataSet *data_obs = new RooDataSet(dataset.c_str(), dataset.c_str(), *w->set("observables")); 
      data_obs->add(*w->set("observables"));
      w->import(*data_obs);
    }
  }

  if (verbose < 0) {
      RooMsgService::instance().setStreamStatus(0,kFALSE);
      RooMsgService::instance().setStreamStatus(1,kFALSE);
      RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
  }

  const RooArgSet * observables = w->set("observables");
  const RooArgSet * nuisances   = w->set("nuisances");

  if (!isnan(rMin_)) w->var("r")->setMin(rMin_);
  if (!isnan(rMax_)) w->var("r")->setMax(rMax_);

  w->saveSnapshot("clean", w->allVars());

  if (nToys == 0) { // observed (usually it's the Asimov data set)
    RooAbsData *dobs = w->data(dataset.c_str());
    if (dobs == 0) {
      std::cerr << "No observed data '" << dataset << "' in the workspace. Cannot compute limit.\n" << std::endl;
      return;
    }
    std::cout << "Computing limit starting from observation" << std::endl;
    utils::printRAD(dobs);
    if (mklimit(w,*dobs,limit)) tree->Fill();
  }
  
  
  std::vector<double> limitHistory;
  if (nToys > 0) {
    double expLimit = 0;
    unsigned int nLimits = 0;
    w->loadSnapshot("clean");
    RooDataSet *systDs = 0;
    if (withSystematics && (readToysFromHere == 0)) {
      if (nuisances == 0 || w->pdf("nuisancePdf") == 0) {
        std::cerr << "ERROR: nuisances or nuisancePdf not set. Perhaps you wanted to run with no systematics?\n" << std::endl;
        abort();
      }
      systDs = w->pdf("nuisancePdf")->generate(*nuisances, nToys);
    }
    for (iToy = 1; iToy <= nToys; ++iToy) {
      RooAbsData *absdata_toy = 0;
      if (readToysFromHere == 0) {
	w->loadSnapshot("clean");
	utils::printPdf(w, "model_b");
	if (withSystematics) {
	  std::auto_ptr<RooArgSet> vars(w->pdf("model_b")->getVariables());
	  *vars = *systDs->get(iToy-1);
	  utils::printPdf(w, "model_b");
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
	absdata_toy = dynamic_cast<RooAbsData *>(readToysFromHere->Get(TString::Format("toy_%d",iToy)));
	if (absdata_toy == 0) {
	  std::cerr << "Toy toy_"<<iToy<<" not found in " << readToysFromHere->GetName() << ". List follows:\n";
	  readToysFromHere->ls();
	  return;
	}
      }
      utils::printRAD(absdata_toy);
      w->loadSnapshot("clean");
      utils::printPdf(w, "model_b");
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
    if (verbose > 0)
      cout << "mean   expected limit: r < " << expLimit << " @ " << cl*100 << "%CL (" <<nLimits << " toyMC)" << endl;
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


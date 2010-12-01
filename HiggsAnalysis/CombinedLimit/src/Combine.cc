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
#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"

using namespace RooStats;
using namespace RooFit;

LimitAlgo * algo;

Float_t t_cpu_, t_real_;
TDirectory *writeToysHere = 0;
TDirectory *sprnwriteToysHere = 0;
TDirectory *readToysFromHere = 0;
bool verbose;
bool withSystematics;
float cl;

void printRDH(RooDataHist *data) {
  std::vector<std::string> varnames, catnames;
  const RooArgSet *b0 = data->get(0);
  TIterator *iter = b0->createIterator();
  for (RooAbsArg *a = 0; (a = (RooAbsArg *)iter->Next()) != 0; ) {
    if (a->InheritsFrom("RooRealVar")) {
      varnames.push_back(a->GetName());
    } else if (a->InheritsFrom("RooCategory")) {
      catnames.push_back(a->GetName());
    }
  }
  delete iter;
  size_t nv = varnames.size(), nc = catnames.size();
  printf(" bin  ");
  for (size_t j = 0; j < nv; ++j) { printf("%10.10s  ", varnames[j].c_str()); }
  for (size_t j = 0; j < nc; ++j) { printf("%10.10s  ", catnames[j].c_str()); }
  printf("  weight\n");
  for (int i = 0, nb = data->numEntries(); i < nb; ++i) {
    const RooArgSet *bin = data->get(i);
    printf("%4d  ",i);
    for (size_t j = 0; j < nv; ++j) { printf("%10g  ",    bin->getRealValue(varnames[j].c_str())); }
    for (size_t j = 0; j < nc; ++j) { printf("%10.10s  ", bin->getCatLabel(catnames[j].c_str())); }
    printf("%8.3f\n", data->weight(*bin,0));
  }
}

void printRAD(const RooAbsData *d) {
  if (d->InheritsFrom("RooDataHist")) printRDH((RooDataHist*)d);
  else d->get(0)->Print("V");
}

void printPdf(RooWorkspace *w, const char *pdfName) {
  std::cout << "PDF " << pdfName << " parameters." << std::endl;
  std::auto_ptr<RooArgSet> params(w->pdf("model_b")->getVariables());
  params->Print("V");
}

bool mklimit(RooWorkspace *w, RooAbsData &data, double &limit) {
  TStopwatch timer;
  bool ret = false;
  try {
    ret = algo->run(w, data, limit);    
  } catch (std::exception &ex) {
    std::cerr << "Caught exception " << ex.what() << std::endl;
    return false;
  }
  if ((ret == false) && verbose) {
    std::cout << "Failed for method " << algo->name() << "\n";
    std::cout << "  --- DATA ---\n";
    printRAD(&data);
    std::cout << "  --- MODEL ---\n";
    w->Print("V");
  }
  timer.Stop(); t_cpu_ = timer.CpuTime()/60.; t_real_ = timer.RealTime()/60.;
  printf("Done in %.2f min (cpu), %.2f min (real)\n", t_cpu_, t_real_);
  return ret;
}

void doCombination(TString hlfFile, const std::string &dataset, double &limit, int &iToy, TTree *tree, int nToys) {
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
      if (verbose) std::cout << "CMSSW_BASE is set, so will try to get include dir for roofit from scram." << std::endl;
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
                  if (verbose) std::cout << "Adding " << buff << " to include path" << std::endl;
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

  /*
    RooMsgService::instance().setStreamStatus(0,kFALSE);
    RooMsgService::instance().setStreamStatus(1,kFALSE);
    RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
  */
  const RooArgSet * observables = w->set("observables");
  const RooArgSet * nuisances   = w->set("nuisances");
  w->saveSnapshot("clean", w->allVars());
  if (nToys == 0) { // observed (usually it's the Asimov data set)
    RooAbsData *dobs = w->data(dataset.c_str());
    if (dobs == 0) {
      std::cerr << "No observed data '" << dataset << "' in the workspace. Cannot compute limit.\n" << std::endl;
      return;
    }
    std::cout << "Computing limit starting from observation" << std::endl;
    printRAD(dobs);
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
	printPdf(w, "model_b");
	if (withSystematics) {
	  RooArgSet *vars = w->pdf("model_b")->getVariables();
	  *vars = *systDs->get(iToy-1);
	  delete vars;
	  printPdf(w, "model_b");
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
      printRAD(absdata_toy);
      w->loadSnapshot("clean");
      printPdf(w, "model_b");
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
    if (verbose)
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
}


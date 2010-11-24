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

#include <RooStats/BayesianCalculator.h>
#include <RooStats/FeldmanCousins.h>
#include <RooStats/HLFactory.h>
#include <RooStats/HybridCalculatorOriginal.h>
#include <RooStats/HypoTestInverter.h>
#include <RooStats/HypoTestInverterPlot.h>
#include <RooStats/HypoTestInverterResult.h>
#include <RooStats/LikelihoodInterval.h>
#include <RooStats/LikelihoodIntervalPlot.h>
#include <RooStats/MCMCCalculator.h>
#include <RooStats/MCMCInterval.h>
#include <RooStats/ModelConfig.h>
#include <RooStats/ModelConfig.h>
#include <RooStats/PointSetInterval.h>
#include <RooStats/ProfileLikelihoodCalculator.h>
#include <RooStats/ProposalHelper.h>
#include <RooStats/UniformProposal.h>
#include <RooStats/SimpleInterval.h>

using namespace RooStats;

TCanvas *c1 = new TCanvas("c1", "c1");

MethodType method = undefined;

Float_t t_cpu_, t_real_;
//RooWorkspace *writeToysHere = 0;
TDirectory *writeToysHere = 0;
TDirectory *readToysFromHere = 0;

void printRAD(const RooAbsData *d) ;
bool mklimit_h2(RooWorkspace *w, RooAbsData &data, double &limit, bool withSystematics=true, bool verbose=false) ;
bool mklimit_pl(RooWorkspace *w, RooAbsData &data, double &limit, bool withSystematics=true, bool verbose=false) ;
bool mklimit_bf(RooWorkspace *w, RooAbsData &data, double &limit, bool withSystematics=true, bool verbose=false) ;
bool mklimit_mcmc(RooWorkspace *w, RooAbsData &data, double &limit, bool uniformProposal, bool withSystematics=true, bool verbose=false); 
void combine(RooWorkspace *w, const std::string &dataset, double &limit, int &iToy, TTree *tree, int nToys=0, bool withSystematics=true);

void doCombination(TString hlfFile, const std::string &dataset, double &limit, int &iToy, TTree *tree, int nToys, bool withSystematics) {
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
  combine(w, dataset, limit, iToy, tree, nToys, withSystematics);
}

bool mklimit(RooWorkspace *w, RooAbsData &data, double &limit, bool withSystematics=true, bool verbose=false) {
  TStopwatch timer;
  bool ret = false;
  try {
    if      (method == hybrid)            ret = mklimit_h2(w,data,limit,withSystematics,verbose);
    else if (method == profileLikelihood) ret = mklimit_pl(w,data,limit,withSystematics,verbose);
    else if (method == bayesianFlatPrior) ret = mklimit_bf(w,data,limit,withSystematics,verbose);
    else if (method == mcmc)              ret = mklimit_mcmc(w,data,limit,false,withSystematics,verbose);
    else if (method == mcmcUniform)       ret = mklimit_mcmc(w,data,limit,true ,withSystematics,verbose);
    else std::cerr << "Unknown method " << method << std::endl;
  } catch (std::exception &ex) {
    std::cerr << "Caught exception " << ex.what() << std::endl;
    return false;
  }
  if ((ret == false) && verbose) {
    std::cout << "Failed for method " << method << "\n";
    std::cout << "  --- DATA ---\n";
    printRAD(&data);
    std::cout << "  --- MODEL ---\n";
    w->Print("V");
  }
  timer.Stop(); t_cpu_ = timer.CpuTime()/60.; t_real_ = timer.RealTime()/60.;
  printf("Done in %.2f min (cpu), %.2f min (real)\n", t_cpu_, t_real_);
  return ret;
}

bool mklimit_pl(RooWorkspace *w, RooAbsData &data, double &limit, bool withSystematics, bool verbose) {
  RooRealVar *r = w->var("r");
  RooArgSet  poi(*r);
  double rMax = r->getMax();
  for (;;) {
    ProfileLikelihoodCalculator plcB(data, *w->pdf("model_s"), poi);
    plcB.SetConfidenceLevel(0.95);
    LikelihoodInterval* plInterval = plcB.GetInterval();
    if (plInterval == 0) return false;
    limit = plInterval->UpperLimit(*r); 
    delete plInterval;
    if (limit >= 0.75*r->getMax()) { 
      std::cout << "Limit r < " << limit << "; r max < " << r->getMax() << std::endl;
      if (r->getMax()/rMax > 20) return false;
      r->setMax(r->getMax()*2); 
      continue;
    }
    if (verbose) {
      /*
	ProfileLikelihoodCalculator plcS(data, *w->pdf("model_s"), poi);
	RooArgSet nullParamValues; 
	nullParamValues.addClone(*r); ((RooRealVar&)nullParamValues["r"]).setVal(0);
	plcS.SetNullParameters(nullParamValues);
	double plSig = plcS.GetHypoTest()->Significance();
      */
      
      std::cout << "\n -- Profile Likelihood -- " << "\n";
      std::cout << "Limit: r < " << limit << " @ 95% CL" << std::endl;
      //std::cout << "Significance: " << plSig << std::endl;
      
    }
    break;
  }
  return true;
}

bool mklimit_bf(RooWorkspace *w, RooAbsData &data, double &limit, bool withSystematics, bool verbose) {
  RooRealVar *r = w->var("r");
  RooUniform  flatPrior("flatPrior","flatPrior",*r);
  RooArgSet  poi(*r);
  double rMax = r->getMax();
  for (;;) {
    BayesianCalculator bcalc(data, *w->pdf("model_s"), poi, flatPrior, (withSystematics ? w->set("nuisances") : 0));
    bcalc.SetLeftSideTailFraction(0);
    bcalc.SetConfidenceLevel(0.95); 
    SimpleInterval* bcInterval = bcalc.GetInterval();
    if (bcInterval == 0) return false;
    limit = bcInterval->UpperLimit();
    if (limit >= 0.75*r->getMax()) { 
      std::cout << "Limit r < " << limit << "; r max < " << r->getMax() << std::endl;
      if (r->getMax()/rMax > 20) return false;
      r->setMax(r->getMax()*2); 
      continue;
    }
    std::cout << "\n -- Bayesian, flat prior -- " << "\n";
    std::cout << "Limit: r < " << limit << " @ 95% CL" << std::endl;
    if (0 && verbose) {
      RooPlot *bcPlot = bcalc.GetPosteriorPlot(true,0.1); 
      bcPlot->Draw(); 
      c1->Print("plots/bc_plot.png");
    }
    break;
  }
  return true;
}

bool mklimit_mcmc(RooWorkspace *w, RooAbsData &data, double &limit, bool uniformProposal, bool withSystematics, bool verbose) {
  RooRealVar *r = w->var("r");
  RooArgSet  poi(*r);
  RooArgSet const &obs = *w->set("observables");
  
  RooUniform  flatPrior("flatPrior","flatPrior",*r);
  RooFitResult *fit = w->pdf("model_s")->fitTo(data, RooFit::Save());
  if (fit == 0) { std::cerr << "Fit failed." << std::endl; return false; }
  fit->Print("V");
  w->loadSnapshot("clean");

  if (withSystematics && (w->set("nuisances") == 0)) {
    std::cerr << "ERROR: nuisances not set. Perhaps you wanted to run with no systematics?\n" << std::endl;
    abort();
  }
  
  ModelConfig modelConfig("sb_model", w);
  modelConfig.SetPdf(*w->pdf("model_s"));
  modelConfig.SetObservables(obs);
  modelConfig.SetParametersOfInterest(poi);
  if (withSystematics) modelConfig.SetNuisanceParameters(*w->set("nuisances"));
  
  ProposalHelper ph;
  ph.SetVariables((RooArgSet&)fit->floatParsFinal());
  ph.SetCovMatrix(fit->covarianceMatrix());
  ph.SetUpdateProposalParameters(true);
  ph.SetCacheSize(100);
  ProposalFunction* pdfProp = ph.GetProposalFunction();  // that was easyA
  if (uniformProposal) { pdfProp = new UniformProposal(); } // might do this in a cleaner way in the future
  
  MCMCCalculator mc(data, modelConfig);
  mc.SetNumIters( 200000 ); 
  mc.SetConfidenceLevel(0.95);
  mc.SetNumBurnInSteps( 500 ); 
  mc.SetProposalFunction(*pdfProp);
  mc.SetNumBins (1000) ; // bins to use for RooRealVars in histograms
  mc.SetLeftSideTailFraction(0);
  mc.SetPriorPdf(flatPrior);
  
  MCMCInterval* mcInt = (MCMCInterval*)mc.GetInterval(); 
  if (mcInt == 0) return false;
  limit = mcInt->UpperLimit(*r);
  std::cout << "\n -- MCMC, flat prior -- " << "\n";
  std::cout << "Limit: r < " << limit << " @ 95% CL" << std::endl;
  std::cout << "Interval:    [ " << mcInt->LowerLimit(*r)             << " , " << mcInt->UpperLimit(*r)             << " ] @ 90% CL" << std::endl;
  std::cout << "Interval DH: [ " << mcInt->LowerLimitByDataHist(*r)   << " , " << mcInt->UpperLimitByDataHist(*r)   << " ] @ 90% CL" << std::endl;
  std::cout << "Interval H:  [ " << mcInt->LowerLimitByHist(*r)       << " , " << mcInt->UpperLimitByHist(*r)       << " ] @ 90% CL" << std::endl;
  //std::cout << "Interval K:  [ " << mcInt->LowerLimitByKeys(*r)       << " , " << mcInt->UpperLimitByKeys(*r)       << " ] @ 90% CL" << std::endl;
  std::cout << "Interval S:  [ " << mcInt->LowerLimitShortest(*r)     << " , " << mcInt->UpperLimitShortest(*r)     << " ] @ 90% CL" << std::endl;
  std::cout << "Interval TF: [ " << mcInt->LowerLimitTailFraction(*r) << " , " << mcInt->UpperLimitTailFraction(*r) << " ] @ 90% CL" << std::endl;
  return true;
}

bool mklimit_h2(RooWorkspace *w, RooAbsData &data, double &limit, bool withSystematics, bool verbose) {
  RooRealVar *r = w->var("r"); r->setConstant(true);
  RooArgSet  poi(*r);
  w->loadSnapshot("clean");
  RooAbsPdf *altModel  = w->pdf("model_s"), *nullModel = w->pdf("model_b");
  
  HybridCalculatorOriginal* hc = new HybridCalculatorOriginal(data,*altModel,*nullModel);
  if (withSystematics) {
    if ((w->set("nuisances") == 0) || (w->pdf("nuisancePdf") == 0)) {
          std::cerr << "ERROR: nuisances or nuisancePdf not set. Perhaps you wanted to run with no systematics?\n" << std::endl;
          abort();
    }
    hc->UseNuisance(true);
    hc->SetNuisancePdf(*w->pdf("nuisancePdf"));
    hc->SetNuisanceParameters(*w->set("nuisances"));
  } else {
    hc->UseNuisance(false);
  }
  hc->SetTestStatistic(1); // 3 = TeV
  hc->PatchSetExtended(false); // Number counting, each dataset has 1 entry 
  hc->SetNumberOfToys(500);
  
  double clsTarget = 0.05, clsAcc  = 0.005, rAcc = 0.1, rRelAcc = 0.05; 
  double clsMin = 1, clsMax = 0, clsMinErr = 0, clsMaxErr = 0;
  double rMin   = 0, rMax = r->getMax();
  
  std::cout << "Search for upper limit to the limit" << std::endl;
  HybridResult *hcResult = 0;
  for (;;) {
    r->setVal(r->getMax()); hcResult = hc->GetHypoTest();
    std::cout << "r = " << r->getVal() << ": CLs = " << hcResult->CLs() << " +/- " << hcResult->CLsError() << std::endl;
    if (hcResult->CLs() == 0) break;
    r->setMax(r->getMax()*2);
    if (r->getVal()/rMax >= 20) { 
      std::cerr << "Cannot set higher limit: at r = " << r->getVal() << " still get CLs = " << hcResult->CLs() << std::endl;
      return false;
    }
  }
  rMax = r->getMax();
  
  std::cout << "Now doing proper bracketing & bisection" << std::endl;
  do {
    r->setVal(0.5*(rMin+rMax));
    hcResult = hc->GetHypoTest();
    if (hcResult == 0) {
      std::cerr << "Hypotest failed" << std::endl;
      return false;
    }
    double clsMid = hcResult->CLs(), clsMidErr = hcResult->CLsError();
    std::cout << "r = " << r->getVal() << ": CLs = " << clsMid << " +/- " << clsMidErr << std::endl;
    while (fabs(clsMid-clsTarget) < 3*clsMidErr && clsMidErr >= clsAcc) {
      HybridResult *more = hc->GetHypoTest();
      hcResult->Add(more);
      clsMid    = hcResult->CLs(); 
      clsMidErr = hcResult->CLsError();
      std::cout << "r = " << r->getVal() << ": CLs = " << clsMid << " +/- " << clsMidErr << std::endl;
    }
    if (verbose) {
      std::cout << "r = " << r->getVal() << ": \n" <<
	"\tCLs      = " << hcResult->CLs()      << " +/- " << hcResult->CLsError()      << "\n" <<
	"\tCLb      = " << hcResult->CLb()      << " +/- " << hcResult->CLbError()      << "\n" <<
	"\tCLsplusb = " << hcResult->CLsplusb() << " +/- " << hcResult->CLsplusbError() << "\n" <<
	std::endl;
    }
    if (fabs(clsMid-clsTarget) <= clsAcc) {
      std::cout << "reached accuracy." << std::endl;
      break;
    }
    if ((clsMid>clsTarget) == (clsMax>clsTarget)) {
      rMax = r->getVal(); clsMax = clsMid; clsMaxErr = clsMidErr;
    } else {
      rMin = r->getVal(); clsMin = clsMid; clsMinErr = clsMidErr;
    }
  } while (rMax-rMin > std::max(rAcc, rRelAcc * r->getVal()));
  if (clsMinErr == 0 && clsMaxErr == 0) {
    std::cerr << "Error: both boundaries have no passing/failing toys.\n";
    return false;
  }
  limit = r->getVal();
  std::cout << "\n -- HypoTestInverter -- \n";
  std::cout << "Limit: r < " << limit << " +/- " << (rMax - rMin) << "\n";
  
  return true;
}

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

RooWorkspace *mksimpleModel(int channels, double nS[], double nB[], double dS, double dB[], bool withSystematics=true) {
  std::cout << "Preparing combination of " << channels << " channels.\n";
  for (int i = 0; i < channels; ++i) {
    printf("Channel %2d: nS = %8.4f,  nB = %8.4f,  dB/B = %6.4f\n", i, nS[i], nB[i], (withSystematics ? dB[i] : 0));
  }
  if (withSystematics) std::cout << "Common stematical uncertainty on signal: " << dS << "\n";
  
  RooWorkspace *w = new RooWorkspace("w","w");
  w->factory("r[1,0,30]");
  w->factory("Gaussian::thetaSignalPdf(thetaSignal[-5,5], 0, 1)");
  
  RooArgSet  observables, nuisances;
  RooArgSet  dataValues_s, dataValues_b;
  TString npoissonians_s = "", npoissonians_b = "", ngaussians="thetaSignalPdf,", nobservables = "", nnuisances = "";
  for (int ich = 0; ich < channels; ++ich) {
    TString sch = TString::Format("%d", ich);
    w->factory("n_obs"+sch+"[0, 10000]"); 
    observables.add(*w->var("n_obs"+sch)); nobservables += TString(ich  ? "," : "")+"n_obs"+sch;
    dataValues_s.addClone(*w->var("n_obs"+sch)); ((RooRealVar &)dataValues_s["n_obs"+sch]).setVal(nS[ich]+nB[ich]);
    dataValues_b.addClone(*w->var("n_obs"+sch)); ((RooRealVar &)dataValues_b["n_obs"+sch]).setVal( 0     +nB[ich]);
    
    if (withSystematics) {
      w->factory(TString::Format("Gaussian::theta%dPdf(theta%d[-5,5], 0, 1)", ich, ich));
      nuisances.add(*w->var("theta"+sch)); nnuisances += TString(ich  ? "," : "")+"theta"+sch;
      ngaussians += TString(ich  ? "," : "")+"theta"+sch+"Pdf";
      
      w->factory(TString::Format("expr::b_exp_%d ('%g*pow(%f,theta%d)', thetaSignal, theta%d)",                  ich, nB[ich], 1.+dB[ich], ich, ich));
      w->factory(TString::Format("expr::sb_exp_%d('r*%g*pow(%f,thetaSignal)+b_exp_%d',r,thetaSignal,b_exp_%d)",  ich, nS[ich], 1.+dS,      ich, ich));
    } else {
      w->factory(TString::Format("b_exp_%d[%g]", ich, nB[ich]));
      w->factory(TString::Format("expr::sb_exp_%d('r*%g+b_exp_%d',r, b_exp_%d)", ich, nS[ich], ich, ich));
    }
    
    w->factory(TString::Format("Poisson::counting%d_s(n_obs%d, sb_exp_%d)", ich, ich, ich));
    w->factory(TString::Format("Poisson::counting%d_b(n_obs%d,  b_exp_%d)", ich, ich, ich));
    npoissonians_s += TString(ich ? "," : "")+"counting"+sch+"_s";
    npoissonians_b += TString(ich ? "," : "")+"counting"+sch+"_b";
  }
  w->factory("set::poi(r)");
  w->factory("set::observables("+nobservables+")");
  if (withSystematics) {
    w->factory("set::nuisances("+nnuisances+")");
    w->factory("PROD::nuisancePdf("+ngaussians+")");
    w->factory("PROD::model_obs_s("+npoissonians_s+")");
    w->factory("PROD::model_obs_b("+npoissonians_b+")");
    w->factory("PROD::model_s(model_obs_s,nuisancePdf)");
    w->factory("PROD::model_b(model_obs_b,nuisancePdf)");
  } else {
    w->factory("PROD::model_s("+npoissonians_s+")");
    w->factory("PROD::model_b("+npoissonians_b+")");
  }
  
  //w->Print("v");
  
  // Observed data (Asimov, in this case)
  RooDataSet *data_obs = new RooDataSet("data_obs", "data_obs", observables); 
  data_obs->add(dataValues_b);
  w->import(*data_obs);
  
  return w;
}

struct SimpleShapeChannel { 
  std::vector<double> nS;
  std::vector<double> nB;
  double dB; 
};

RooWorkspace *mksimpleShapeModel(const std::vector<SimpleShapeChannel> &channels, double dS,  bool withSystematics=true) 
{
  std::cout << "Preparing combination of " << channels.size() << " channels.\n";
  for (int i = 0, n = channels.size(); i < n; ++i) {
    const SimpleShapeChannel &ch = channels[i];
    assert(ch.nS.size() == ch.nB.size());
    printf("Channel %2d: %2d bins, dB/B = %6.4f\n", i, int(ch.nS.size()), (withSystematics ? ch.dB : 0));
    for (int j = 0, nj = ch.nS.size(); j < nj; ++j) {
      printf("  bin %2d: nS = %8.4f,  nB = %8.4f\n", j, ch.nS[j], ch.nB[j]);
    }
  }
  if (withSystematics) std::cout << "Common stematical uncertainty on signal: " << dS << "\n";
  
  RooWorkspace *w = new RooWorkspace("w","w");
  w->factory("r[1,0,30]");
  w->factory("Gaussian::thetaSignalPdf(thetaSignal[-5,5], 0, 1)");
  
  RooArgSet  observables;
  RooArgSet  dataValues_s, dataValues_b;
  TString npoissonians_s = "", npoissonians_b = "", ngaussians="thetaSignalPdf,", nobservables = "", nnuisances = "";
  for (size_t ich = 0; ich < channels.size(); ++ich) {
    const SimpleShapeChannel &ch = channels[ich];
    assert(ch.nS.size() == ch.nB.size());
    TString sch = TString::Format("%d", ich);
    
    if (withSystematics) {
      w->factory(TString::Format("Gaussian::theta%dPdf(theta%d[-5,5], 0, 1)", ich, ich));
      nnuisances += TString(ich  ? "," : "")+"theta"+sch;
      ngaussians += TString(ich  ? "," : "")+"theta"+sch+"Pdf";
    }
    
    for (size_t  j = 0; j < ch.nS.size(); ++j) {
      TString schb = TString::Format("%d_%d", ich,j);
      
      // observable for bin j of channel ich
      w->factory("n_obs"+schb+"[0, 10000]"); 
      observables.add(*w->var("n_obs"+schb)); 
      nobservables += TString(ich+j  ? "," : "")+"n_obs"+schb;
      
      // Asimov yields for S+B and B    
      dataValues_s.addClone(*w->var("n_obs"+schb)); 
      dataValues_b.addClone(*w->var("n_obs"+schb)); 
      ((RooRealVar &)dataValues_s["n_obs"+schb]).setVal(ch.nS[j]+ch.nB[j]);
      ((RooRealVar &)dataValues_b["n_obs"+schb]).setVal( 0      +ch.nB[j]);
      
      // expected yield
      if (withSystematics) {
	w->factory(TString::Format("expr::b_exp_%d_%d   ('%g * pow(  %f,   theta%d)', thetaSignal, theta%d)",                  
				   ich, j, ch.nB[j],   1.+ch.dB,    ich,                   ich));
	w->factory(TString::Format("expr::sb_exp_%d_%d('r*%g  *  pow( %f,thetaSignal)+b_exp_%d_%d',r,thetaSignal,b_exp_%d_%d)",  
				   ich,j, ch.nS[j],   1.+dS,                  ich,j,                     ich,j));
      } else {
	w->factory(TString::Format("b_exp_%d_%d[%g]", ich, j, ch.nB[j]));
	w->factory(TString::Format("expr::sb_exp_%d_%d('r*%g+b_exp_%d_%d',r, b_exp_%d_%d)", ich,j, ch.nS[j], ich,j, ich,j));
      }
      
      // poissonians    
      w->factory(TString::Format("Poisson::counting%d_%d_s(n_obs%d_%d, sb_exp_%d_%d)", ich,j, ich,j,  ich,j));
      w->factory(TString::Format("Poisson::counting%d_%d_b(n_obs%d_%d,  b_exp_%d_%d)", ich,j, ich,j,  ich,j));
      npoissonians_s += TString(ich+j ? "," : "")+"counting"+schb+"_s";
      npoissonians_b += TString(ich+j ? "," : "")+"counting"+schb+"_b";
    }
  }
  w->factory("set::poi(r)");
  w->factory("set::observables("+nobservables+")");
  if (withSystematics) {
    w->factory("set::nuisances("+nnuisances+")");
    w->factory("PROD::nuisancePdf("+ngaussians+")");
    w->factory("PROD::model_obs_s("+npoissonians_s+")");
    w->factory("PROD::model_obs_b("+npoissonians_b+")");
    w->factory("PROD::model_s(model_obs_s,nuisancePdf)");
    w->factory("PROD::model_b(model_obs_b,nuisancePdf)");
  } else {
    w->factory("PROD::model_s("+npoissonians_s+")");
    w->factory("PROD::model_b("+npoissonians_b+")");
  }
  
  //w->Print("v");
  
  // Observed data (Asimov, in this case)
  RooDataSet *data_obs = new RooDataSet("data_obs", "data_obs", observables); 
  data_obs->add(dataValues_b);
  w->import(*data_obs);
  
  return w;
}

void combine(RooWorkspace *w, const std::string &dataset, double &limit, int &iToy, TTree *tree, int nToys, bool withSystematics) {
  using namespace RooFit;
  using namespace RooStats;
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
    if (mklimit(w,*dobs,limit,withSystematics,true)) tree->Fill();
  }
  
  if (nToys > 0) {
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
      if (mklimit(w,*absdata_toy,limit,withSystematics,true)) tree->Fill();
      if (writeToysHere) {
	//writeToysHere->import(*absdata_toy, RooFit::Rename(TString::Format("toy_%d", iToy)), RooFit::Silence());
	writeToysHere->WriteTObject(absdata_toy, TString::Format("toy_%d", iToy));
      }
      delete absdata_toy;
    }
  }
}

void combine(int channels, double nS[], double nB[], double dS, double dB[], 
             double &limit, int &iToy, TTree *tree, int nToys=0, bool withSystematics=true) {
  RooWorkspace *w = mksimpleModel(channels, nS, nB, dS, dB, withSystematics);
  combine(w, "data_obs", limit, iToy, tree, nToys, withSystematics);
}

void combine(const std::vector<SimpleShapeChannel> &channels, double dS, 
             double &limit, int &iToy, TTree *tree, int nToys=0, bool withSystematics=true) {
  RooWorkspace *w = mksimpleShapeModel(channels, dS, withSystematics);
  combine(w, "data_obs", limit, iToy, tree, nToys, withSystematics);
}

bool readChannel(const char *fileName, int mass, double &nS, double &nB, double &dB) {
  FILE *f = fopen(fileName, "r");
  if (f == 0) {
    std::cout << "  file " << fileName << " does not exist or is not readable." << std::endl;
    return false;
  }
  int test_mass; 
  while (fscanf(f,"%d %lf %lf %lf\n", &test_mass, &nS, &nB, &dB) != EOF) {
    if (test_mass == mass) break;
  }
  fclose(f);
  if (test_mass == mass) {
    std::cout << "  channel " << fileName << " for mH =  " << mass << " got S = " << nS << ", B = " << nB << ", dB/B = " << dB << std::endl;
    return true;
  } else {
    std::cout << "  channel " << fileName << " does not contain mH =  " << mass << std::endl;
    return false;
  }
}

bool readChannelShape(const char *fileName, int mass, std::vector<SimpleShapeChannel> &out) {
  FILE *f = fopen(fileName, "r");
  if (f == 0) {
    std::cout << "  file " << fileName << " does not exist or is not readable." << std::endl;
    return false;
  }
  int test_mass; char buff[1024];
  std::vector<double> vals;
  while (fscanf(f,"%d", &test_mass) != EOF) {
    fgets(buff,1024,f); // read the rest of the line into buff
    if (test_mass != mass) continue;
    
    // now read the N doubles in buff, put them in vals
    vals.clear();
    char *ptr, *saveptr;
    ptr = strtok_r(buff, " \t", &saveptr);
    if (ptr == 0) { printf("Error parsing mass line %d of file  %s\n",test_mass, fileName); return false; }
    do {
      errno = 0; // reset the error flag
      vals.push_back(atof(ptr));
      if (errno != 0) {  printf("Error parsing token %s in mass line %d of file %s\n", ptr, test_mass, fileName); return false; }
      ptr = strtok_r(NULL, " \t", &saveptr);
    } while (ptr);
    
    if (vals.size() % 2 != 1) {
      printf("Error: mass line %d of filename %s does not conform to the (<S> <B>)+ <dB> format\n",test_mass,fileName); 
      return false;
    }
    out.push_back(SimpleShapeChannel());
    SimpleShapeChannel & ch = out.back();
    ch.dB = vals.back();
    for (size_t i = 0, n = vals.size()-1; i < n; i += 2) {
      ch.nS.push_back(vals[i]  );
      ch.nB.push_back(vals[i+1]);
    }
    break;
  }
  fclose(f);
  if (test_mass == mass) {
    std::cout << "  channel " << fileName << " for mH =  " << mass << " got " << out.back().nS.size() << " bins " << std::endl;
    for (int j = 0, nj = out.back().nS.size(); j < nj; ++j) {
      printf("       bin %2d: nS = %8.4f,  nB = %8.4f\n", j, out.back().nS[j], out.back().nB[j]);
    }
    return true;
  } else {
    std::cout << "  channel " << fileName << " does not contain mH =  " << mass << std::endl;
    return false;
  }
}


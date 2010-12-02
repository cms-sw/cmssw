#include "HiggsAnalysis/CombinedLimit/interface/utils.h"

#include <cstdio>
//#include <cerrno>
#include <iostream>
//#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
//#include <exception>
//#include <algorithm>

//#include <TCanvas.h>
//#include <TFile.h>
//#include <TGraphErrors.h>
#include <TIterator.h>
//#include <TLine.h>
//#include <TMath.h>
//#include <TString.h>
//#include <TSystem.h>
//#include <TStopwatch.h>
//#include <TTree.h>

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
//#include <RooFitResult.h>
//#include <RooMsgService.h>
//#include <RooPlot.h>
//#include <RooRandom.h>
#include <RooRealVar.h>
//#include <RooUniform.h>
#include <RooWorkspace.h>

//#include <RooStats/HLFactory.h>
//#include <boost/filesystem.hpp>
//#include <boost/program_options.hpp>

void utils::printRDH(RooDataHist *data) {
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

void utils::printRAD(const RooAbsData *d) {
  if (d->InheritsFrom("RooDataHist")) printRDH((RooDataHist*)d);
  else d->get(0)->Print("V");
}

void utils::printPdf(RooWorkspace *w, const char *pdfName) {
  std::cout << "PDF " << pdfName << " parameters." << std::endl;
  std::auto_ptr<RooArgSet> params(w->pdf("model_b")->getVariables());
  params->Print("V");
}

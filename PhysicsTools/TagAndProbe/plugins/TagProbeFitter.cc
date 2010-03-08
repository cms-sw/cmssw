#include "PhysicsTools/TagAndProbe/interface/TagProbeFitter.h"
//#include "TagProbeFitter.h"

#include "TROOT.h"
#include "TFile.h"
#include "TPad.h"
#include "TText.h"
#include "TCanvas.h"
#include "RooWorkspace.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"
#include "RooAddPdf.h"
#include "RooGlobalFunc.h"
#include "RooCategory.h"
#include "RooSimultaneous.h"
#include "RooPlot.h"
#include "RooFitResult.h"
#include "RooBinning.h"
#include "RooBinningCategory.h"
#include "RooMultiCategory.h"
#include "RooMappedCategory.h"
#include "Roo1DTable.h"
#include "RooMinuit.h"
#include "RooNLLVar.h"

using namespace RooFit;

TagProbeFitter::TagProbeFitter(vector<string> inputFileNames, string inputDirectoryName, string inputTreeName, string outputFileName, int numCPU_, bool saveWorkspace_){
  inputTree = new TChain((inputDirectoryName+"/"+inputTreeName).c_str());
  for(size_t i=0; i<inputFileNames.size(); i++){
    inputTree->Add(inputFileNames[i].c_str());
  }
  outputFile = new TFile(outputFileName.c_str(),"recreate");
  outputDirectory = outputFile->mkdir(inputDirectoryName.c_str());
  numCPU = numCPU_;
  saveWorkspace = saveWorkspace_;
}

TagProbeFitter::~TagProbeFitter(){
  if(inputTree)
    delete inputTree;
  if(outputFile)
    outputFile->Close();
}

bool TagProbeFitter::addVariable(string name, string title, double low, double hi, string units){
  variables.addClone(RooRealVar(name.c_str(), title.c_str(), low, hi, units.c_str()));
  return true;
}

bool TagProbeFitter::addCategory(string name, string title, string expression){
  RooCategory* c = (RooCategory*) parameterParser.factory(expression.c_str());
  if(!c)
    return false;
  //set the name of the category to the passed name instead of the one in the expression
  c->SetName(name.c_str());
  c->SetTitle(title.c_str());
  variables.addClone(*c);
  return true;
}

void TagProbeFitter::addPdf(string name, vector<string>& pdfCommands){
  pdfs[name] = pdfCommands;
}

string TagProbeFitter::calculateEfficiency(string dirName, string effCat, string effState, vector<string>& unbinnedVariables, map<string, vector<double> >& binnedReals, map<string, std::vector<string> >& binnedCategories, vector<string>& binToPDFmap, bool saveWorkspace){
  //go to home directory
  outputDirectory->cd();
  //make a directory corresponding to this efficiency binning
  gDirectory->mkdir(dirName.c_str())->cd();

  //collect unbinned variables
  RooArgSet unbinnedVars;
  for(vector<string>::iterator v=unbinnedVariables.begin(); v!=unbinnedVariables.end(); v++){
    unbinnedVars.addClone(variables[v->c_str()]);
  }
  //collect the binned variables and the corresponding bin categories
  RooArgSet binnedVariables;
  RooArgSet binCategories;
  for(map<string, vector<double> >::iterator v=binnedReals.begin(); v!=binnedReals.end(); v++){
    TString name = v->first;
    binnedVariables.addClone(variables[name]);
    ((RooRealVar&)binnedVariables[name]).setBinning( RooBinning(v->second.size()-1, &v->second[0]) );
    binCategories.addClone( RooBinningCategory(name+"_bins", name+"_bins", (RooRealVar&)binnedVariables[name]) );
  }
  //collect the category variables and the corresponding mapped categories
  RooArgSet categories;
  RooArgSet mappedCategories;
  for(map<string, vector<string> >::iterator v=binnedCategories.begin(); v!=binnedCategories.end(); v++){
    TString name = v->first;
    categories.addClone(variables[name]);
    mappedCategories.addClone(RooMappedCategory(name+"_bins", name+"_bins", (RooCategory&)categories[name]));
    for(uint i = 0; i<v->second.size(); i++){
      ((RooMappedCategory&)mappedCategories[name+"_bins"]).map(v->second[i].c_str(), name+"_"+TString(v->second[i].c_str()).ReplaceAll(",","_"));
    }
  }

  //now add the necessary mass and passing variables to make the unbinned RooDataSet
  RooDataSet data("data", "data", inputTree, RooArgSet( RooArgSet(binnedVariables, categories), RooArgSet(unbinnedVars, variables[effCat.c_str()]) ));
  //merge the bin categories to a MultiCategory for convenience
  RooMultiCategory allCats("allCats", "allCats", RooArgSet(binCategories, mappedCategories));
  data.addColumn(allCats);
  //setup the efficiency category
  RooMappedCategory efficiencyCategory("_efficiencyCategory_", "_efficiencyCategory_", (RooCategory&)variables[effCat.c_str()], "Failed");
  efficiencyCategory.map(effState.c_str(), "Passed");
  data.addColumn( efficiencyCategory );
  //setup the pdf category
  RooMappedCategory pdfCategory("_pdfCategory_", "_pdfCategory_", allCats, (binToPDFmap.size()>0)?binToPDFmap[0].c_str():"");
  for(uint i = 1; i<binToPDFmap.size(); i+=2){
    pdfCategory.map(binToPDFmap[i].c_str(), binToPDFmap[i+1].c_str());
  }
  data.addColumn( pdfCategory );

  //create the empty efficiency datasets from the binned variables
  RooRealVar efficiency("efficiency", "Efficiency", 0, 1);
  RooDataSet fitEfficiency("fit_eff", "Efficiency from unbinned ML fit", RooArgSet(RooArgSet(binnedVariables, categories), efficiency), StoreAsymError(RooArgSet(binnedVariables, efficiency)));
  RooDataSet sbsEfficiency("sbs_eff", "Efficiency from side band substraction", RooArgSet(RooArgSet(binnedVariables, categories), efficiency), StoreAsymError(RooArgSet(binnedVariables, efficiency)));
  RooDataSet cntEfficiency("cnt_eff", "Efficiency from counting", RooArgSet(RooArgSet(binnedVariables, categories), efficiency), StoreAsymError(RooArgSet(binnedVariables, efficiency)));

  //loop over all bins with the help of allCats
  TIterator* it = allCats.typeIterator();
  for(RooCatType* t = (RooCatType*)it->Next(); t!=0; t = (RooCatType*)it->Next() ){
    //name of the multicategory
    TString catName = t->GetName();
    //skip unmapped states
    if(catName.Contains("NotMapped")) continue;
    //create the dataset
    RooDataSet* data_bin = (RooDataSet*) data.reduce(//SelectVars(RooArgSet(variables["mass"], variables["passing"])),
      Cut(TString::Format("allCats==%d",t->getVal())));
    //set the category variables by reading the first event
    const RooArgSet* row = data_bin->get();
    //make directory name
    TString dirName = catName;
    dirName.ReplaceAll("{","").ReplaceAll("}","").ReplaceAll(";","__");
    if(TString(((RooCategory&)(*row)["_pdfCategory_"]).getLabel()).Length() > 0){
      dirName.Append("__").Append(((RooCategory&)(*row)["_pdfCategory_"]).getLabel());
    }
    cout<<"Fitting bin:  "<<dirName<<endl;
    //make a directory for each bin
    gDirectory->mkdir(dirName)->cd();
    //create a workspace
    RooWorkspace* w = new RooWorkspace();
    //import the data
    w->import(*data_bin);
    //svae the distribution of variables
    saveDistributionsPlot(w);
    //do the fitting only if there is sufficient number of events
    if(data_bin->numEntries()>0){
      //set the values of binnedVariables to the mean value in this data bin
      RooArgSet meanOfVariables;
      TIterator* vit = binnedVariables.createIterator();
      for(RooRealVar* v = (RooRealVar*)vit->Next(); v!=0; v = (RooRealVar*)vit->Next() ){
        meanOfVariables.addClone(*v);
        double mean = w->data("data")->mean(*v);
        RooBinning binning((RooBinning&)v->getBinning());
        int ind = binning.binNumber(mean);
        RooRealVar& newVar = (RooRealVar&)meanOfVariables[v->GetName()];
        newVar.setVal(mean);
        newVar.setAsymError(binning.binLow(ind)-mean, binning.binHigh(ind)-mean);
      }
      //put an entry in the efficiency dataset
      //note that the category values are coming from data_bin->get(0)
      efficiency.setVal(0);//reset
      efficiency.setAsymError(0,0);
      doFitEfficiency(w, pdfCategory.getLabel(), efficiency);
      fitEfficiency.add( RooArgSet(RooArgSet(meanOfVariables, *data_bin->get(0)), efficiency) );

      efficiency.setVal(0);//reset
      doSBSEfficiency(w, efficiency);
      sbsEfficiency.add( RooArgSet(RooArgSet(meanOfVariables, *data_bin->get(0)), efficiency) );

      efficiency.setVal(0);//reset
      doCntEfficiency(w, efficiency);
      cntEfficiency.add( RooArgSet(RooArgSet(meanOfVariables, *data_bin->get(0)), efficiency) );
    }
    //save the workspace if requested
    if(saveWorkspace){
      w->Write("w");
    }
    //clean up
    delete data_bin;
    delete w;
    //get back to the initial directory
    gDirectory->cd("..");
  }
  //save the efficiency data
  fitEfficiency.Write();
  gDirectory->mkdir("fit_eff_plots")->cd();
  saveEfficiencyPlots(fitEfficiency, effCat+"::"+effState, binnedVariables, mappedCategories);
  gDirectory->cd("..");

  sbsEfficiency.Write();
  gDirectory->mkdir("sbs_eff_plots")->cd();
  saveEfficiencyPlots(sbsEfficiency, effCat+"::"+effState, binnedVariables, mappedCategories);
  gDirectory->cd("..");

  cntEfficiency.Write();
  gDirectory->mkdir("cnt_eff_plots")->cd();
  saveEfficiencyPlots(cntEfficiency, effCat+"::"+effState, binnedVariables, mappedCategories);
  gDirectory->cd("..");
  //empty string means no error
  return "";
}

void TagProbeFitter::doFitEfficiency(RooWorkspace* w, string pdfName, RooRealVar& efficiency){
  //if pdfName is empty skip the fit
  if(pdfName == ""){
    return;
  }
  //create the simultaneous pdf of name pdfName
  createPdf(w, pdfs[pdfName]);
  //set the initial values for the yields of signal and background
  setInitialValues(w);
  //do the fit and get hold of the FitResults
  //RooFitResult* res = w->pdf("simPdf")->fitTo(*w->data("data"), Save());
  RooNLLVar nll("nll", "nll", *w->pdf("simPdf"), *w->data("data"), Extended(), NumCPU(numCPU));
  RooMinuit m(nll);
  m.setErrorLevel(0.5);
  m.setStrategy(2);
  m.hesse();
  m.migrad();
  m.hesse();
  m.minos(*w->var("efficiency"));
  RooFitResult* res = m.save();
  // save everything
  res->Write("fitresults");
  w->saveSnapshot("finalState",w->components());
  saveFitPlot(w);
  //extract the efficiency parameter from the results
  RooRealVar* e = (RooRealVar*) res->floatParsFinal().find("efficiency");
  //What's wrong with this? and why don't they copy the errors!
  //efficiency = *e;
  efficiency.setVal(e->getVal());
  efficiency.setAsymError(e->getErrorLo(), e->getErrorHi());
}

void TagProbeFitter::createPdf(RooWorkspace* w, vector<string>& pdfCommands){
  // create the signal and background pdfs defined by the user
  for(uint i=0; i<pdfCommands.size(); i++){
    w->factory(pdfCommands[i].c_str());
  }
  // setup the simultaneous extended pdf
  w->factory("expr::numSignalPass('efficiency*numSignalAll', efficiency, numSignalAll[0.,1e10])");
  w->factory("expr::numSignalFail('(1-efficiency)*numSignalAll', efficiency, numSignalAll)");
  w->factory("SUM::pdfPass(numSignalPass*signal, numBackgroundPass[0,1e10]*backgroundPass)");
  w->factory("SUM::pdfFail(numSignalFail*signal, numBackgroundFail[0,1e10]*backgroundFail)");
  w->factory("SIMUL::simPdf(_efficiencyCategory_, Passed=pdfPass, Failed=pdfFail)");
  // signalFractionInPassing is not used in the fit just to set the initial values
  if(w->var("signalFractionInPassing") == 0)
    w->factory("signalFractionInPassing[0.9]");
}

void TagProbeFitter::setInitialValues(RooWorkspace* w){
  // calculate initial values
  double signalEfficiency = w->var("efficiency")->getVal();
  double signalFractionInPassing = w->var("signalFractionInPassing")->getVal();
  double totPassing = w->data("data")->sumEntries("_efficiencyCategory_==_efficiencyCategory_::Passed");
  double totFailinging = w->data("data")->sumEntries("_efficiencyCategory_==_efficiencyCategory_::Failed");
  double numSignalAll = totPassing*signalFractionInPassing/signalEfficiency;
  // check if this value is inconsistent on the failing side
  if(numSignalAll*(1-signalEfficiency) > totFailinging)
    numSignalAll = totFailinging;
  // now set the values
  w->var("numSignalAll")->setVal(numSignalAll);
  w->var("numBackgroundPass")->setVal(totPassing - numSignalAll*signalEfficiency);
  w->var("numBackgroundFail")->setVal(totFailinging -  numSignalAll*(1-signalEfficiency));
  // save initial state for reference
  w->saveSnapshot("initialState",w->components());
}

void TagProbeFitter::saveFitPlot(RooWorkspace* w){
  // save refferences for convenience
  RooRealVar& mass = *w->var("mass");
  RooCategory& efficiencyCategory = *w->cat("_efficiencyCategory_");
  RooAbsData& data = *w->data("data");
  RooAbsPdf& pdf = *w->pdf("simPdf");
  // make a 2x2 canvas
  TCanvas canvas("fit_canvas");
  canvas.Divide(2,2);
  // plot the Passing Probes
  canvas.cd(1);
  RooPlot* passFrame = mass.frame(Name("Passing"), Title("Passing Probes"));
  data.plotOn(passFrame, Cut("_efficiencyCategory_==_efficiencyCategory_::Passed"));
  pdf.plotOn(passFrame, Slice(efficiencyCategory, "Passed"), ProjWData(efficiencyCategory, data), LineColor(kGreen));
  pdf.plotOn(passFrame, Slice(efficiencyCategory, "Passed"), ProjWData(efficiencyCategory, data), LineColor(kGreen), Components("backgroundPass"), LineStyle(kDashed));
  passFrame->Draw();
  // plot the Failing Probes
  canvas.cd(2);
  RooPlot* failFrame = mass.frame(Name("Failing"), Title("Failing Probes"));
  data.plotOn(failFrame, Cut("_efficiencyCategory_==_efficiencyCategory_::Failed"));
  pdf.plotOn(failFrame, Slice(efficiencyCategory, "Failed"), ProjWData(efficiencyCategory, data), LineColor(kRed));
  pdf.plotOn(failFrame, Slice(efficiencyCategory, "Failed"), ProjWData(efficiencyCategory, data), LineColor(kRed), Components("backgroundFail"), LineStyle(kDashed));
  failFrame->Draw();
  // plot the All Probes
  canvas.cd(3);
  RooPlot* allFrame = mass.frame(Name("All"), Title("All Probes"));
  data.plotOn(allFrame);
  pdf.plotOn(allFrame, ProjWData(efficiencyCategory, data), LineColor(kBlue));
  pdf.plotOn(allFrame, ProjWData(efficiencyCategory, data), LineColor(kBlue), Components("backgroundPass,backgroundFail"), LineStyle(kDashed));
  allFrame->Draw();
  // plot the Parameters
  canvas.cd(4);
  RooPlot* statFrame = mass.frame(Name("Parameters"), Title("Parameters"));
  pdf.paramOn(statFrame, &data);
  statFrame->Draw();
  //save and clean up
  canvas.Write();
  delete passFrame;
  delete failFrame;
  delete allFrame;
  delete statFrame;
}

void TagProbeFitter::saveDistributionsPlot(RooWorkspace* w){
  // save refferences for convenience
  RooAbsData& data = *w->data("data");

  const RooArgSet* vars = data.get();
  vector<RooRealVar*> reals;
  TIterator* it = vars->createIterator();
  for(RooAbsArg* v = (RooAbsArg*)it->Next(); v!=0; v = (RooAbsArg*)it->Next() ){
    if(!v->InheritsFrom("RooRealVar")) continue;
    reals.push_back((RooRealVar*)v);
  }

  TCanvas canvas("distributions_canvas");
  canvas.Divide(3,reals.size());
  for(uint i=0; i<reals.size(); i++){
    canvas.cd(3*i+1);
    RooPlot* passFrame = reals[i]->frame(Name("Passing"), Title("Passing Probes"), Bins(100));
    data.plotOn(passFrame, Cut("_efficiencyCategory_==_efficiencyCategory_::Passed"), LineColor(kGreen));
    data.statOn(passFrame);
    passFrame->Draw();
    canvas.cd(3*i+2);
    RooPlot* failFrame = reals[i]->frame(Name("Failing"), Title("Failing Probes"), Bins(100));
    data.plotOn(failFrame, Cut("_efficiencyCategory_==_efficiencyCategory_::Failed"), LineColor(kRed));
    data.statOn(failFrame);
    failFrame->Draw();
    // plot the All Probes
    canvas.cd(3*i+3);
    RooPlot* allFrame = reals[i]->frame(Name("All"), Title("All Probes"), Bins(100));
    data.plotOn(allFrame, LineColor(kBlue));
    data.statOn(allFrame);
    allFrame->Draw();
  }
  canvas.Write();
}

void TagProbeFitter::saveEfficiencyPlots(RooDataSet& eff, string effName, RooArgSet& binnedVariables, RooArgSet& mappedCategories){
  const RooArgSet* set = eff.get();
  RooRealVar* e = (RooRealVar*) set->find("efficiency");
  TIterator* plotVarIt = binnedVariables.createIterator();
  for(RooRealVar* plotVar = (RooRealVar*)plotVarIt->Next(); plotVar!=0; plotVar = (RooRealVar*)plotVarIt->Next() ){
    RooArgSet binCategories;
    TIterator* it = binnedVariables.createIterator();
    for(RooRealVar* v = (RooRealVar*)it->Next(); v!=0; v = (RooRealVar*)it->Next() ){
      if(plotVar==v) continue;
      binCategories.addClone( RooBinningCategory(TString(v->GetName())+"_bins", TString(v->GetName())+"_bins", *v) );
    }
    RooMultiCategory allCats("allCats", "allCats", RooArgSet(binCategories, mappedCategories));
    RooDataSet myEff(eff);
    myEff.addColumn(allCats);
    TIterator* catIt = allCats.typeIterator();
    for(RooCatType* t = (RooCatType*)catIt->Next(); t!=0; t = (RooCatType*)catIt->Next() ){
      TString catName = t->GetName();
      if(catName.Contains("NotMapped")) continue;
      catName.ReplaceAll("{","").ReplaceAll("}","").ReplaceAll(";","__");
      RooDataSet* data_bin = (RooDataSet*) myEff.reduce( Cut(TString::Format("allCats==%d",t->getVal())) );
      RooPlot* p = plotVar->frame(Name(TString(plotVar->GetName())+"_plot__"+catName), Title(catName));
      data_bin->plotOnXY(p,YVar(*e));
      p->SetYTitle(TString("Efficiency of ")+effName.c_str());
      p->SetAxisRange(0,1,"Y");
      p->Write();
      delete p;
      delete data_bin;
    }
  }
}

void TagProbeFitter::doSBSEfficiency(RooWorkspace* w, RooRealVar& efficiency){
}

void TagProbeFitter::doCntEfficiency(RooWorkspace* w, RooRealVar& efficiency){
  int pass = w->data("data")->sumEntries("_efficiencyCategory_==_efficiencyCategory_::Passed");
  int fail = w->data("data")->sumEntries("_efficiencyCategory_==_efficiencyCategory_::Failed");
  double e, lo, hi;
  //from TGraphAsymmErrors
  Efficiency( pass, pass+fail, 0.683, e, lo, hi );
  efficiency.setVal(e);
  efficiency.setAsymError(lo-e, hi-e);
}

int main(int argc, char* argv[]){
  vector<string> inputs;
  inputs.push_back("testNewWrite.root");
  TagProbeFitter f(inputs, "MakeHisto", "fitter_tree", "myplots.root", 1, true);

  f.addVariable("mass", "tag-probe mass", 2.5, 3.8, "GeV/c^{2}");
  f.addVariable("pt", "probe pT", 3., 10., "GeV/c");
  f.addVariable("eta", "probe #eta", -2.1, 2.1, "");
  f.addCategory("mcTrue", "MC True", "c[true=1,false=0]");
  f.addCategory("passing", "isMuon", "c[pass=1,fail=0]");

  vector<string> pdfCommands;
  pdfCommands.push_back("efficiency[0.9,0,1]");
  pdfCommands.push_back("Gaussian::signal(mass, mean[3.1,3.0,3.2], sigma[0.03,0.01,0.05])");
  pdfCommands.push_back("Chebychev::backgroundPass(mass, cPass[0,-1,1])");
  pdfCommands.push_back("Chebychev::backgroundFail(mass, cFail[0,-1,1])");
  f.addPdf("g", pdfCommands);
  vector<string> pdfCommands2;
  pdfCommands2.push_back("efficiency[0.9,0,1]");
  pdfCommands2.push_back("Gaussian::signal(mass, mean[3.1,3.0,3.2], sigma[0.03,0.01,0.05])");
  pdfCommands2.push_back("Chebychev::backgroundPass(mass, {cPass[0,-1,1], cPass2[0,-1,1]})");
  pdfCommands2.push_back("Chebychev::backgroundFail(mass, {cFail[0,-1,1], cFail2[0,-1,1]})");
  f.addPdf("g2", pdfCommands2);

  vector<string> unbinnedVariables;
  unbinnedVariables.push_back("mass");

  map<string, vector<double> >binnedReals;
  vector<double> ptBins;
  ptBins.push_back(3.);
  ptBins.push_back(4.5);
  ptBins.push_back(8.);
  ptBins.push_back(10.);
  binnedReals["pt"] = ptBins;
  vector<double> etaBins;
  etaBins.push_back(-2.1);
  etaBins.push_back(-1.2);
  etaBins.push_back(0.);
  etaBins.push_back(1.2);
  etaBins.push_back(2.1);
  binnedReals["eta"] = etaBins;

  map<string, vector<string> > binnedStates;

  vector<string> binToPDFmap;
  binToPDFmap.push_back("g");
  binToPDFmap.push_back("*pt_bin0*");
  binToPDFmap.push_back("g2");

  f.calculateEfficiency("pt_eta", "passing", "pass", unbinnedVariables, binnedReals, binnedStates, binToPDFmap, true);

  vector<string> mcBins;
  mcBins.push_back("true");
  binnedStates["mcTrue"] = mcBins;

  vector<string> emptyMap; //no fits here

  f.calculateEfficiency("pt_eta_mcTrue", "passing", "pass", unbinnedVariables, binnedReals, binnedStates, emptyMap, true);
}

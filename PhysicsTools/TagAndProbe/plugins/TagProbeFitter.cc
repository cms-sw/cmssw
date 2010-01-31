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

TagProbeFitter::TagProbeFitter(string inputFileName, string inputDirectoryName, string inputTreeName, string outputFileName, bool saveWorkspace_){
  inputFile = new TFile(inputFileName.c_str());
  inputFile->Cd(inputDirectoryName.c_str());
  inputTree = (TTree*)gROOT->FindObject(inputTreeName.c_str());
  outputFile = new TFile(outputFileName.c_str(),"recreate");
  outputDirectory = outputFile->mkdir(inputDirectoryName.c_str());
  saveWorkspace = saveWorkspace_;
  RooCategory passing("passing","passing");
  passing.defineType("pass",1);
  passing.defineType("fail",0);
  variables.addClone(passing);
}

TagProbeFitter::~TagProbeFitter(){
  if(inputFile)
    inputFile->Close();
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

string TagProbeFitter::calculateEfficiency(string dirName, string pdfName, map<string, vector<double> >& binning, map<string, std::vector<string> >& states, bool saveWorkspace){
  //check if the pdfExists
  if(pdfs[pdfName]==vector<string>())
    return string("The specified PDF: ")+pdfName+" was not defined!";
  //go to home directory
  outputDirectory->cd();
  //make a directory corresponding to this efficiency binning
  gDirectory->mkdir(dirName.c_str())->cd();

  //collect the binned variables and the corresponding bin categories
  RooArgSet binnedVariables;
  RooArgSet binCategories;
  for(map<string, vector<double> >::iterator v=binning.begin(); v!=binning.end(); v++){
    TString name = v->first;
    binnedVariables.addClone(variables[name]);
    ((RooRealVar&)binnedVariables[name]).setBinning( RooBinning(v->second.size()-1, &v->second[0]) );
    binCategories.addClone( RooBinningCategory(name+"_bins", name+"_bins", (RooRealVar&)binnedVariables[name]) );
  }
  //collect the category variables and the corresponding mapped categories
  RooArgSet categories;
  RooArgSet mappedCategories;
  for(map<string, vector<string> >::iterator v=states.begin(); v!=states.end(); v++){
    TString name = v->first;
    categories.addClone(variables[name]);
    mappedCategories.addClone(RooMappedCategory(name+"_bins", name+"_bins", (RooCategory&)categories[name]));
    for(uint i = 0; i<v->second.size(); i++){
      ((RooMappedCategory&)mappedCategories[name+"_bins"]).map(v->second[i].c_str(), TString::Format("%s_bin%d", name.Data(), i));
    }
  }

  //create the empty efficiency datasets from the binned variables
  RooRealVar efficiency("efficiency","Efficiency",0,1);
  RooDataSet fitEfficiency("fit_eff", "Efficiency from unbinned ML fit", RooArgSet(RooArgSet(binnedVariables, categories), efficiency), StoreAsymError(efficiency));
  RooDataSet sbsEfficiency("sbs_eff", "Efficiency from side band substraction", RooArgSet(RooArgSet(binnedVariables, categories), efficiency), StoreAsymError(efficiency));
  RooDataSet cntEfficiency("cnt_eff", "Efficiency from counting", RooArgSet(RooArgSet(binnedVariables, categories), efficiency), StoreAsymError(efficiency));
  //now add the necessary mass and passing variables to make the unbinned RooDataSet
  RooDataSet data("data", "data", inputTree, RooArgSet( RooArgSet(binnedVariables, categories), RooArgSet(variables["mass"], variables["passing"]) ));
  //merge the bin categories to a MultiCategory for convenience
  RooMultiCategory allCats("allCats", "allCats", RooArgSet(binCategories, mappedCategories));
  //hook up the categories
  data.addColumn(allCats);

  //loop over all bins with the help of allCats
  TIterator* it = allCats.typeIterator();
  for(RooCatType* t = (RooCatType*)it->Next(); t!=0; t = (RooCatType*)it->Next() ){
    //name of the multicategory
    TString catName = t->GetName();
    //skip unmapped states
    if(catName.Contains("NotMapped")) continue;
    //make directory name
    TString dirName = catName;
    dirName.ReplaceAll("{","").ReplaceAll("}","").ReplaceAll(";","-");
    //make a directory for each bin
    gDirectory->mkdir(dirName)->cd();
    //create the dataset
    RooDataSet* data_bin = (RooDataSet*) data.reduce(//SelectVars(RooArgSet(variables["mass"], variables["passing"])),
      Cut(TString::Format("allCats==%d",t->getVal())));
    //create a workspace
    RooWorkspace* w = new RooWorkspace();
    //import the data
    w->import(*data_bin);
    //do the fitting only if there is sufficient number of events
    if(data_bin->numEntries()>0){
      //set the values of binnedVariables to the mean value in this data bin
      TIterator* vit = binnedVariables.createIterator();
      for(RooRealVar* v = (RooRealVar*)vit->Next(); v!=0; v = (RooRealVar*)vit->Next() ){
        v->setVal(w->data("data")->mean(*v));
      }
      //put an entry in the efficiency dataset
      //note that the category values are coming from data_bin->get(0)
      efficiency.setVal(0);//reset
      doFitEfficiency(w, pdfName, efficiency);
      fitEfficiency.add( RooArgSet(RooArgSet(binnedVariables, *data_bin->get(0)), efficiency) );

      efficiency.setVal(0);//reset
      doSBSEfficiency(w, efficiency);
      sbsEfficiency.add( RooArgSet(RooArgSet(binnedVariables, *data_bin->get(0)), efficiency) );

      efficiency.setVal(0);//reset
      doCntEfficiency(w, efficiency);
      cntEfficiency.add( RooArgSet(RooArgSet(binnedVariables, *data_bin->get(0)), efficiency) );
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
  sbsEfficiency.Write();
  cntEfficiency.Write();
  //empty string means no error
  return "";
}

void TagProbeFitter::doFitEfficiency(RooWorkspace* w, string pdfName, RooRealVar& efficiency){
  //create the simultaneous pdf of name pdfName
  createPdf(w, pdfs[pdfName]);
  //set the initial values for the yields of signal and background
  setInitialValues(w);
  //do the fit and get hold of the FitResults
  //RooFitResult* res = w->pdf("simPdf")->fitTo(*w->data("data"), Save());
  RooNLLVar nll("nll", "nll", *w->pdf("simPdf"), *w->data("data"), kTRUE);
  RooMinuit m(nll);
  m.setErrorLevel(0.5);
  m.setStrategy(2);
  m.hesse();
  m.migrad();
  m.hesse();
  m.minos();
  RooFitResult* res = m.save();
  // save everything
  res->Write("fitresults");
  w->saveSnapshot("finalState",w->components());
  savePlot(w);
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
  w->factory("SIMUL::simPdf(passing, pass=pdfPass, fail=pdfFail)");
  // signalFractionInPassing is not used in the fit just to set the initial values
  if(w->var("signalFractionInPassing") == 0)
    w->factory("signalFractionInPassing[0.9]");
}

void TagProbeFitter::setInitialValues(RooWorkspace* w){
  // calculate initial values
  double signalEfficiency = w->var("efficiency")->getVal();
  double signalFractionInPassing = w->var("signalFractionInPassing")->getVal();
  double totPassing = w->data("data")->sumEntries("passing==1");
  double totFailinging = w->data("data")->sumEntries("passing==0");
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

void TagProbeFitter::savePlot(RooWorkspace* w){
  // save refferences for convenience
  RooRealVar& mass = *w->var("mass");
  RooCategory& passing = *w->cat("passing");
  RooAbsData& data = *w->data("data");
  RooAbsPdf& pdf = *w->pdf("simPdf");
  // make a 2x2 canvas
  TCanvas canvas("fit_canvas");
  canvas.Divide(2,2);
  // plot the Passing Probes
  canvas.cd(1);
  RooPlot* passFrame = mass.frame(Name("Passing"), Title("Passing Probes"));
  data.plotOn(passFrame,Cut("passing==passing::pass"),LineColor(kGreen));
  pdf.plotOn(passFrame,Slice(passing,"pass"),ProjWData(passing,data),LineColor(kGreen));
  passFrame->Draw();
  // plot the Failing Probes
  canvas.cd(2);
  RooPlot* failFrame = w->var("mass")->frame(Name("Failing"), Title("Failing Probes"));
  data.plotOn(failFrame,Cut("passing==passing::fail"),LineColor(kRed));
  pdf.plotOn(failFrame,Slice(passing,"fail"),ProjWData(passing,data),LineColor(kRed));
  failFrame->Draw();
  // plot the All Probes
  canvas.cd(3);
  RooPlot* allFrame = mass.frame(Name("All"), Title("All Probes"));
  data.plotOn(allFrame);
  pdf.plotOn(allFrame,ProjWData(passing,data),LineColor(kBlue));
  pdf.paramOn(allFrame); //temporarily plot on this frame
  allFrame->Draw();
  // plot the Parameters
  canvas.cd(4);
  RooPlot* statFrame = mass.frame(Name("Parameters"), Title("Parameters"));
//    simPdf.paramOn(statFrame); why doesn't this work?!
  data.statOn(statFrame);
  statFrame->Draw();
  //save and clean up
  canvas.Write();
  delete passFrame;
  delete failFrame;
  delete allFrame;
  delete statFrame;
}

void TagProbeFitter::doSBSEfficiency(RooWorkspace* w, RooRealVar& efficiency){
}

void TagProbeFitter::doCntEfficiency(RooWorkspace* w, RooRealVar& efficiency){
  int pass = w->data("data")->sumEntries("passing==passing::pass");
  int fail = w->data("data")->sumEntries("passing==passing::fail");
  double e, lo, hi;
  //from TGraphAsymmErrors
  Efficiency( pass, pass+fail, 0.683, e, lo, hi );
  efficiency.setVal(e);
  efficiency.setAsymError(lo-e, hi-e);
}

int main(int argc, char* argv[]){
  TagProbeFitter f("testNewWrite.root", "MakeHisto", "fitter_tree", "myplots.root", true);

  f.addVariable("mass", "tag-probe mass", 2.5, 3.8, "GeV/c^{2}");
  f.addVariable("pt", "probe pT", 3., 10., "GeV/c");
  f.addVariable("eta", "probe #eta", -2.1, 2.1, "");
  f.addCategory("mcTrue", "MC True", "c[true=1,false=0]");

  vector<string> pdfCommands;
  pdfCommands.push_back("efficiency[0.9,0,1]");
  pdfCommands.push_back("Gaussian::signal(mass, mean[3.1,3.0,3.2], sigma[0.03,0.01,0.05])");
  pdfCommands.push_back("Chebychev::backgroundPass(mass, cPass[0,-1,1])");
  pdfCommands.push_back("Chebychev::backgroundFail(mass, cFail[0,-1,1])");
  f.addPdf("g", pdfCommands);

  map<string, vector<double> >bins;
  vector<double> ptBins;
  ptBins.push_back(3.);
  ptBins.push_back(5.);
  ptBins.push_back(10.);
  bins["pt"] = ptBins;
  vector<double> etaBins;
  etaBins.push_back(-2.1);
  etaBins.push_back(-1.2);
  etaBins.push_back(0.);
  etaBins.push_back(1.2);
  etaBins.push_back(2.1);
  bins["eta"] = etaBins;

  map<string, vector<string> > states;
  f.calculateEfficiency("pt_eta", "g", bins, states, true);

  vector<string> mcBins;
  mcBins.push_back("true");
  states["mcTrue"] = mcBins;
  f.calculateEfficiency("pt_eta_mcTrue", "g", bins, states, true);
}

#include "PhysicsTools/TagAndProbe/interface/TagProbeFitter.h"
//#include "TagProbeFitter.h"

#include "TROOT.h"
#include "TFile.h"
#include "TPad.h"
#include "TText.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TStyle.h"
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
#include "RooAbsDataStore.h"
#include "RooEfficiency.h"
#include "RooGaussian.h"
#include "RooChebychev.h"
#include "RooProdPdf.h"
#include "RooGenericPdf.h"
#include "RooExtendPdf.h"
#include "RooTrace.h"

using namespace RooFit;

TagProbeFitter::TagProbeFitter(vector<string> inputFileNames, string inputDirectoryName, string inputTreeName, string outputFileName, int numCPU_, bool saveWorkspace_, bool floatShapeParameters_, std::vector<std::string> fixVars_){
  inputTree = new TChain((inputDirectoryName+"/"+inputTreeName).c_str());
  for(size_t i=0; i<inputFileNames.size(); i++){
    inputTree->Add(inputFileNames[i].c_str());
  }
  outputFile = new TFile(outputFileName.c_str(),"recreate");
  outputDirectory = outputFile->mkdir(inputDirectoryName.c_str());
  numCPU = numCPU_;
  saveWorkspace = saveWorkspace_;
  massBins = 0; // automatic default
  floatShapeParameters = floatShapeParameters_;
  fixVars = fixVars_;
  if(!floatShapeParameters && fixVars.empty()) std::cout << "TagProbeFitter: " << "You wnat to fix some variables but do not specify them!";

  gROOT->SetStyle("Plain");
  gStyle->SetTitleFillColor(0);
  gStyle->SetPalette(1);
  gStyle->SetOptStat(0);
  gStyle->SetPaintTextFormat(".2f");
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

void TagProbeFitter::setBinsForMassPlots(int bins){
  massBins = bins;
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
    for(unsigned int i = 0; i<v->second.size(); i++){
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
  for(unsigned int i = 1; i<binToPDFmap.size(); i+=2){
    pdfCategory.map(binToPDFmap[i].c_str(), binToPDFmap[i+1].c_str());
  }
  data.addColumn( pdfCategory );
  
  //create the empty efficiency datasets from the binned variables
  RooRealVar efficiency("efficiency", "Efficiency", 0, 1);

  RooDataSet fitEfficiency("fit_eff", "Efficiency from unbinned ML fit", RooArgSet(RooArgSet(binnedVariables, categories), efficiency), StoreAsymError(RooArgSet(binnedVariables, efficiency)));
//  RooDataSet sbsEfficiency("sbs_eff", "Efficiency from side band substraction", RooArgSet(RooArgSet(binnedVariables, categories), efficiency), StoreAsymError(RooArgSet(binnedVariables, efficiency)));
  RooDataSet cntEfficiency("cnt_eff", "Efficiency from counting", RooArgSet(RooArgSet(binnedVariables, categories), efficiency), StoreAsymError(RooArgSet(binnedVariables, efficiency)));


  if(!floatShapeParameters){
    //fitting whole dataset to get initial values for some parameters
    RooWorkspace* w = new RooWorkspace();
    w->import(data);
    efficiency.setVal(0);//reset
    efficiency.setAsymError(0,0);
    std::cout << "ALL dataset: calling doFitEfficiency with pdf: " << pdfCategory.getLabel() << std::endl;
    doFitEfficiency(w, pdfCategory.getLabel(), efficiency);
    delete w;
  }

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

    //get PDF name
    TString pdfName(((RooCategory*)row->find("_pdfCategory_"))->getLabel());


    //make directory name
    TString dirName = catName;
    dirName.ReplaceAll("{","").ReplaceAll("}","").ReplaceAll(";","__");
    if(pdfName.Length() > 0){
      dirName.Append("__").Append(pdfName);
    }
    
    cout<<"Fitting bin:  "<<dirName<<endl;
    //make a directory for each bin
    gDirectory->mkdir(dirName)->cd();
    //create a workspace
    RooWorkspace* w = new RooWorkspace();
    //import the data
    w->import(*data_bin);
    //save the distribution of variables
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
      doFitEfficiency(w, pdfName.Data(), efficiency);
      fitEfficiency.add( RooArgSet(RooArgSet(meanOfVariables, *data_bin->get(0)), efficiency) );

/*      efficiency.setVal(0);//reset
      doSBSEfficiency(w, efficiency);
      sbsEfficiency.add( RooArgSet(RooArgSet(meanOfVariables, *data_bin->get(0)), efficiency) );*/

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

/*  sbsEfficiency.Write();
  gDirectory->mkdir("sbs_eff_plots")->cd();
  saveEfficiencyPlots(sbsEfficiency, effCat+"::"+effState, binnedVariables, mappedCategories);
  gDirectory->cd("..");*/

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
  RooFitResult* res;

  //******* The block of code below is to make the fit converge faster.
  // ****** This part is OPTIONAL, i.e., off be default. User can activate this
  // ****** by setting the following parameters: "fixVars" and "floatShapeParameters"
  // ****** Here is the full logic:
  /////   ---> if "floatShapeParameters==true" && "fixVars is empty" : 
  ////             Just perform the fit without any of these optimizations (this is the default logic)
  /////   ---> if "floatShapeParameters==true" && "fixVars is NOT empty" : 
  ////             In each bin, fix the desired parameters and perform the fit to obtain a good starting point.
  /////            Then float these parameters and fit again.
  ////    ---> if "floatShapeParameters==false" && "fixVars is empty" : 
  ////             Do not perform any fit at all. Just print error message.
  ////    ---> if "floatShapeParameters==false" && "fixVars is NOT empty" : 
  ///              Perform a global fit to the whole sample, save the fitted values of the 
  ///              user specified parameters, and fix them for bin-by-bin fit. 

      
  if(!fixVars.empty()){
    // calculate initial values for parameters user want to fix
    if(!floatShapeParameters && fixVarValues.empty()){
      // we want to fix these parameters for each bin.
      // the following sequence fixes them, fits, releases and fits again
      // to get reasonable values. 
      // ----------------------------------------------------------------------
      // This procedure works only once with a whole dataset (without binning)
      // ----------------------------------------------------------------------

      // fix them
      varFixer(w,true);
      //do fit 
      w->pdf("simPdf")->fitTo(*w->data("data"), Save(true), Extended(true), NumCPU(numCPU));
      //release vars
      varFixer(w,false);
      //do fit 
      w->pdf("simPdf")->fitTo(*w->data("data"), Save(true), Extended(true), NumCPU(numCPU));
      //save vars
      varSaver(w);
      // now we have a starting point. Fit will converge faster.
    }
    
    // here we can use initial values if we want (this works for each bin)
    if(!floatShapeParameters) varRestorer(w);  //restore vars

    
    // if we don't want to "floatShapeParameters" we just fix, fit, 
    //  release, and fit again. No need for global fitting above.
    //fix vars
    varFixer(w,true);
    //do fit
    res = w->pdf("simPdf")->fitTo(*w->data("data"), Save(true), Extended(true), NumCPU(numCPU), Minos(true));
  }//if(!fixVars.empty())
  
  // (default = true) if we don't want to fix any parameters or want to fit each bin with all parameters floating
  if(floatShapeParameters){
    //release vars
    varFixer(w,false);
    
    //do fit
    res = w->pdf("simPdf")->fitTo(*w->data("data"), Save(true), Extended(true), NumCPU(numCPU), Minos(true));
  }



//   res = w->pdf("simPdf")->fitTo(*w->data("data"), Save(true), Extended(true), NumCPU(numCPU));

//   RooNLLVar nll("nll", "nll", *w->pdf("simPdf"), *w->data("data"), Extended(), NumCPU(numCPU));
//   RooMinuit m(nll);
//   m.setErrorLevel(0.5);
//   m.setStrategy(2);
//   m.hesse();
//   m.migrad();
//   m.hesse();
//   m.minos(*w->var("efficiency"));
//   RooFitResult* res = m.save();
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
  for(unsigned int i=0; i<pdfCommands.size(); i++){
    w->factory(pdfCommands[i].c_str());
  }
  // setup the simultaneous extended pdf
  w->factory("expr::numSignalPass('efficiency*numSignalAll', efficiency, numSignalAll[0.,1e10])");
  w->factory("expr::numSignalFail('(1-efficiency)*numSignalAll', efficiency, numSignalAll)");
  w->factory("SUM::pdfPass(numSignalPass*signal, numBackgroundPass[0.,1e10]*backgroundPass)");
  w->factory("SUM::pdfFail(numSignalFail*signal, numBackgroundFail[0.,1e10]*backgroundFail)");
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
  RooCategory& efficiencyCategory = *w->cat("_efficiencyCategory_");
  RooAbsData* dataAll = w->data("data");
  RooAbsData* dataPass = dataAll->reduce(Cut("_efficiencyCategory_==_efficiencyCategory_::Passed")); 
  RooAbsData* dataFail = dataAll->reduce(Cut("_efficiencyCategory_==_efficiencyCategory_::Failed")); 
  RooAbsPdf& pdf = *w->pdf("simPdf");
  RooArgSet *obs = pdf.getObservables(*dataAll);
  RooRealVar* mass = 0;
  TIterator* it = obs->createIterator();
  for(RooAbsArg* v = (RooAbsArg*)it->Next(); v!=0; v = (RooAbsArg*)it->Next() ){
    if(!v->InheritsFrom("RooRealVar")) continue;
    mass = (RooRealVar*)v;
    break;
  }
  if(!mass) return;
  // make a 2x2 canvas
  TCanvas canvas("fit_canvas");
  canvas.Divide(2,2);
  vector<RooPlot*> frames;
  // plot the Passing Probes
  canvas.cd(1);
  if (massBins == 0) {
      frames.push_back(mass->frame(Name("Passing"), Title("Passing Probes")));
      frames.push_back(mass->frame(Name("Failing"), Title("Failing Probes")));
      frames.push_back(mass->frame(Name("All"),     Title("All Probes")));
  } else {
      frames.push_back(mass->frame(Name("Passing"), Title("Passing Probes"), Bins(massBins)));
      frames.push_back(mass->frame(Name("Failing"), Title("Failing Probes"), Bins(massBins)));
      frames.push_back(mass->frame(Name("All"),     Title("All Probes"), Bins(massBins)));
  }
  dataPass->plotOn(frames[0]);
  pdf.plotOn(frames[0], Slice(efficiencyCategory, "Passed"), ProjWData(*dataPass), LineColor(kGreen));
  pdf.plotOn(frames[0], Slice(efficiencyCategory, "Passed"), ProjWData(*dataPass), LineColor(kGreen), Components("backgroundPass"), LineStyle(kDashed));
  frames[0]->Draw();
  // plot the Failing Probes
  canvas.cd(2);
  dataFail->plotOn(frames[1]);
  pdf.plotOn(frames[1], Slice(efficiencyCategory, "Failed"), ProjWData(*dataFail), LineColor(kRed));
  pdf.plotOn(frames[1], Slice(efficiencyCategory, "Failed"), ProjWData(*dataFail), LineColor(kRed), Components("backgroundFail"), LineStyle(kDashed));
  frames[1]->Draw();
  // plot the All Probes
  canvas.cd(3);
  dataAll->plotOn(frames.back());
  pdf.plotOn(frames.back(), ProjWData(*dataAll), LineColor(kBlue));
  pdf.plotOn(frames.back(), ProjWData(*dataAll), LineColor(kBlue), Components("backgroundPass,backgroundFail"), LineStyle(kDashed));
  frames.back()->Draw();
  // plot the Fit Results
  canvas.cd(4);
  frames.push_back(mass->frame(Name("Fit Results"), Title("Fit Results")));
  pdf.paramOn(frames.back(), dataAll, "", 0, "NELU", 0.1, 0.9, 0.9);
  // draw only the parameter box not the whole frame
  frames.back()->findObject(Form("%s_paramBox",pdf.GetName()))->Draw();
  //save and clean up
  canvas.Write();
  for (size_t i=0; i<frames.size(); i++) {
    delete frames[i];
  }
  delete dataPass;
  delete dataFail;
}

void TagProbeFitter::saveDistributionsPlot(RooWorkspace* w){
  // save pointers to datasets to manage memory
  RooAbsData* dataAll = w->data("data");
  RooAbsData* dataPass = dataAll->reduce(Cut("_efficiencyCategory_==_efficiencyCategory_::Passed")); 
  RooAbsData* dataFail = dataAll->reduce(Cut("_efficiencyCategory_==_efficiencyCategory_::Failed")); 

  const RooArgSet* vars = dataAll->get();
  vector<RooRealVar*> reals;
  TIterator* it = vars->createIterator();
  for(RooAbsArg* v = (RooAbsArg*)it->Next(); v!=0; v = (RooAbsArg*)it->Next() ){
    if(!v->InheritsFrom("RooRealVar")) continue;
    reals.push_back((RooRealVar*)v);
  }
  TCanvas canvas("distributions_canvas");
  canvas.Divide(3,reals.size());
  vector<RooPlot*> frames;
  for(unsigned int i=0; i<reals.size(); i++){
    // plot the Passing Probes
    canvas.cd(3*i+1);
    frames.push_back(reals[i]->frame(Name("Passing"), Title("Passing Probes"), Bins(100)));
    dataPass->plotOn(frames.back(), LineColor(kGreen));
    dataPass->statOn(frames.back());
    frames.back()->Draw();
    // plot the Failing Probes
    canvas.cd(3*i+2);
    frames.push_back(reals[i]->frame(Name("Failing"), Title("Failing Probes"), Bins(100)));
    dataFail->plotOn(frames.back(), LineColor(kRed));
    dataFail->statOn(frames.back());
    frames.back()->Draw();
    // plot the All Probes
    canvas.cd(3*i+3);
    frames.push_back(reals[i]->frame(Name("All"), Title("All Probes"), Bins(100)));
    dataAll->plotOn(frames.back(), LineColor(kBlue));
    dataAll->statOn(frames.back());
    frames.back()->Draw();
  }
  canvas.Write();
  for (size_t i=0; i<frames.size(); i++) {
    delete frames[i];
  }
  delete dataPass;
  delete dataFail;
}

void TagProbeFitter::saveEfficiencyPlots(RooDataSet& eff, TString effName, RooArgSet& binnedVariables, RooArgSet& mappedCategories){
  TIterator* v1it = binnedVariables.createIterator();
  for(RooRealVar* v1 = (RooRealVar*)v1it->Next(); v1!=0; v1 = (RooRealVar*)v1it->Next() ){
    RooArgSet binCategories1D;
    TIterator* v2it = binnedVariables.createIterator();
    for(RooRealVar* v2 = (RooRealVar*)v2it->Next(); v2!=0; v2 = (RooRealVar*)v2it->Next() ){
      if(v2 == v1) continue;
      binCategories1D.addClone( RooBinningCategory(TString(v2->GetName())+"_bins", TString(v2->GetName())+"_bins", *v2) );

      RooArgSet binCategories2D;
      TIterator* v3it = binnedVariables.createIterator();
      for(RooRealVar* v3 = (RooRealVar*)v3it->Next(); v3!=0; v3 = (RooRealVar*)v3it->Next() ){
        if(v3 == v1 || v3 == v2) continue;
        binCategories2D.addClone( RooBinningCategory(TString(v3->GetName())+"_bins", TString(v3->GetName())+"_bins", *v3) );
      }  
      RooMultiCategory allCats2D("allCats2D", "allCats2D", RooArgSet(binCategories2D, mappedCategories));
      if(allCats2D.numTypes()==0){
        makeEfficiencyPlot2D(eff, *v1, *v2, TString::Format("%s_%s_PLOT", v1->GetName(), v2->GetName()), "", effName);
      }else{
        RooDataSet myEff(eff);
        myEff.addColumn(allCats2D);
        TIterator* catIt = allCats2D.typeIterator();
        for(RooCatType* t = (RooCatType*)catIt->Next(); t!=0; t = (RooCatType*)catIt->Next() ){
          TString catName = t->GetName();
          if(catName.Contains("NotMapped")) continue;
          catName.ReplaceAll("{","").ReplaceAll("}","").ReplaceAll(";","_&_");
          RooDataSet* eff_bin = (RooDataSet*) myEff.reduce( Cut(TString::Format("allCats2D==%d",t->getVal())) );
          makeEfficiencyPlot2D(*eff_bin, *v1, *v2, TString::Format("%s_%s_PLOT_%s",v1->GetName(), v2->GetName(), catName.Data()), catName, effName);
          delete eff_bin;
        }        
      }
    }
    RooMultiCategory allCats1D("allCats1D", "allCats1D", RooArgSet(binCategories1D, mappedCategories));
    if(allCats1D.numTypes()==0){
      makeEfficiencyPlot1D(eff, *v1, TString::Format("%s_PLOT", v1->GetName()), "", effName);
    }else{
      RooDataSet myEff(eff);
      myEff.addColumn(allCats1D);
      TIterator* catIt = allCats1D.typeIterator();
      for(RooCatType* t = (RooCatType*)catIt->Next(); t!=0; t = (RooCatType*)catIt->Next() ){
        TString catName = t->GetName();
        if(catName.Contains("NotMapped")) continue;
        catName.ReplaceAll("{","").ReplaceAll("}","").ReplaceAll(";","_&_");
        RooDataSet* eff_bin = (RooDataSet*) myEff.reduce( Cut(TString::Format("allCats1D==%d",t->getVal())) );
        makeEfficiencyPlot1D(*eff_bin, *v1, TString::Format("%s_PLOT_%s", v1->GetName(), catName.Data()), catName, effName);
        delete eff_bin;
      }
    }
  }
}

void TagProbeFitter::makeEfficiencyPlot1D(RooDataSet& eff, RooRealVar& v, TString plotName, TString plotTitle, TString effName){
  TCanvas canvas(plotName);
  const RooArgSet* set = eff.get();
  RooRealVar* e = (RooRealVar*) set->find("efficiency");
  RooPlot* p = v.frame(Name(plotName), Title(plotTitle));
  eff.plotOnXY(p,YVar(*e));
  p->SetYTitle(TString("Efficiency of ")+effName);
  p->SetAxisRange(0,1,"Y");
  p->Draw();
  canvas.Write();
  delete p;  
}

void TagProbeFitter::makeEfficiencyPlot2D(RooDataSet& eff, RooRealVar& v1, RooRealVar& v2, TString plotName, TString plotTitle, TString effName){
  TCanvas canvas(plotName);
  canvas.SetRightMargin(0.15);
  TH2F* h = new TH2F(plotName, plotName, v1.getBinning().numBins(), v1.getBinning().array(), v2.getBinning().numBins(), v2.getBinning().array());
  const RooArgSet* set = eff.get();
  RooRealVar* e = (RooRealVar*) set->find("efficiency");
  RooRealVar* v1_ = (RooRealVar*) set->find(v1.GetName());
  RooRealVar* v2_ = (RooRealVar*) set->find(v2.GetName());
  h->SetTitle(TString::Format("%s;%s%s;%s%s;Efficiency of %s", plotTitle.Data(),
    v1.GetTitle(), TString(v1.getUnit()).Length()==0?"":TString::Format(" (%s)", v1.getUnit()).Data(),
    v2.GetTitle(), TString(v2.getUnit()).Length()==0?"":TString::Format(" (%s)", v2.getUnit()).Data(), effName.Data()));
  h->SetOption("colztexte");
  h->GetZaxis()->SetRangeUser(-0.001,1.001);
  h->SetStats(kFALSE);
  for(int i=0; i<eff.numEntries(); i++){
    eff.get(i);
    h->SetBinContent(h->FindBin(v1_->getVal(), v2_->getVal()), e->getVal());
    h->SetBinError(h->FindBin(v1_->getVal(), v2_->getVal()), (e->getErrorHi()-e->getErrorLo())/2.);
  }
  h->Draw();
  canvas.Write();
  delete h;
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

void TagProbeFitter::varFixer(RooWorkspace* w, bool fix){
  std::vector<std::string>::const_iterator it;
  for(it=fixVars.begin(); it<fixVars.end(); it++){    
    if(w->var((*it).c_str()))
      w->var((*it).c_str())->setAttribute("Constant",fix);
    else{
      std::cout << "TagProbeFitter: " << "Can't find a variable to fix: " << *it;
    }
  }
}

void TagProbeFitter::varSaver(RooWorkspace* w){
  if(!fixVarValues.empty()){
    std::cout << "attempt to save variables more than once!" << std::endl;
    return;
  }
  std::vector<std::string>::const_iterator it;
  for(it=fixVars.begin(); it<fixVars.end(); it++){
    fixVarValues.push_back(w->var((*it).c_str())->getVal());
  }
  
}

void TagProbeFitter::varRestorer(RooWorkspace* w){
  if(fixVarValues.size()==fixVars.size())
    for(unsigned int i=0; i< fixVars.size(); i++){
      std::cout << "setting variable " << fixVars[i].c_str() << std::endl;
      w->var(fixVars[i].c_str())->setVal(fixVarValues[i]);
    }
  else{
    std::cout << "fixVars and fixVarValues are not of the same size!" << std::endl; 
  }
}


int main(int argc, char* argv[]){
  // create test input file
  TFile out("testTree.root","recreate");
  out.mkdir("Test")->cd();
  // pt
  RooRealVar pt("pt","pt",1,10);
  RooGenericPdf ptPdf("ptPdf","1",pt);
  // eta
  RooRealVar eta("eta","eta",-2.5,2.5);
  // passing
  RooCategory passing("passing","passing");
  passing.defineType("Passed",1);
  passing.defineType("Failed",0);
  // efficiency value
  RooRealVar a("a","a",0.9) ;
  RooRealVar b("b","b",1) ;
  RooRealVar c("c","c",3) ;
//  RooFormulaVar efficiency("efficiency", "a/(1+exp(-b*(pt-c)))", RooArgList(a, b, c, pt)) ;
  RooRealVar efficiency("efficiency", "efficiency", 0.9, 0.0, 1.0);
  RooPlot* curve = pt.frame();
  efficiency.plotOn(curve);
  curve->Write("efficiencyCurve");
  RooEfficiency passingPdf("passingPdf", "passingPdf", efficiency, passing, "Passed");
  // mass
  RooRealVar mass("mass", "mass", 2.6, 3.6);
  RooRealVar mean("mean", "mean", 3.1, 3.0, 3.2);
  RooRealVar sigma("sigma", "sigma", 0.03, 0.01, 0.05);
  RooGaussian signal("signal", "signal", mass, mean, sigma);
  RooRealVar numSignalAll("numSignalAll", "numSignalAll", 1000., 0., 1e10);
  RooExtendPdf signalExt("signalExt", "signalExt", signal, numSignalAll);
  RooProdPdf signalPdf("signalPdf", "signalPdf", RooArgSet(signalExt, ptPdf), Conditional(passingPdf,passing));
  
  RooRealVar cPass("cPass", "cPass", 0.1, -1, 1);
  RooChebychev backgroundPass("backgroundPass", "backgroundPass", mass, cPass);
  RooRealVar numBackgroundPass("numBackgroundPass", "numBackgroundPass", 1000., 0., 1e10);
  RooExtendPdf backgroundPassPdf("backgroundPassPdf", "backgroundPassPdf", backgroundPass, numBackgroundPass);
  
  RooRealVar cFail("cFail", "cFail", 0.1, -1, 1);
  RooChebychev backgroundFail("backgroundFail", "backgroundFail", mass, cFail);
  RooRealVar numBackgroundFail("numBackgroundFail", "numBackgroundFail", 1000., 0., 1e10);
  RooExtendPdf backgroundFailPdf("backgroundFailPdf", "backgroundFailPdf", backgroundFail, numBackgroundFail);
  
  RooSimultaneous backgroundPdf("backgroundPdf", "backgroundPdf", passing);
  backgroundPdf.addPdf(backgroundPassPdf, "Passed");
  backgroundPdf.addPdf(backgroundFailPdf, "Failed");
  RooProdPdf backgroundPtPdf("backgroundPtPdf", "backgroundPtPdf", backgroundPdf, ptPdf);
  // mc_true, True=signal, False=background
  RooCategory mc_true("mc_true","mc_true");
  mc_true.defineType("True",1);
  mc_true.defineType("False",0);

  RooSimultaneous model("model", "model", mc_true);
  model.addPdf(signalPdf, "True");
  model.addPdf(backgroundPtPdf, "False");

  RooDataSet* data = model.generate(RooArgSet(mass, pt, eta, passing, mc_true));
  data->tree()->Write("fitter_tree");
  
  out.Close();
  // end of generation
  
  // fit the test input
  vector<string> inputs;
  inputs.push_back("testTree.root");
  TagProbeFitter f(inputs, "Test", "fitter_tree", "myplots.root", 1, true);

  f.addVariable("mass", "tag-probe mass", 2.6, 3.6, "GeV/c^{2}");
  f.addVariable("pt", "probe pT", 1., 10., "GeV/c");
  f.addVariable("eta", "probe #eta", -2.1, 2.1, "");
  f.addCategory("mc_true_idx", "MC True", "c[true=1,false=0]");
  f.addCategory("passing_idx", "isPassing", "c[pass=1,fail=0]");

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
  ptBins.push_back(1.);
  ptBins.push_back(4.5);
  ptBins.push_back(8.);
  ptBins.push_back(10.);
  binnedReals["pt"] = ptBins;
  vector<double> etaBins;
  etaBins.push_back(-2.5);
//  etaBins.push_back(-1.2);
//  etaBins.push_back(0.);
//  etaBins.push_back(1.2);
  etaBins.push_back(2.5);
  binnedReals["eta"] = etaBins;

  map<string, vector<string> > binnedStates;

  vector<string> binToPDFmap;
  binToPDFmap.push_back("g");
  binToPDFmap.push_back("*pt_bin0*");
  binToPDFmap.push_back("g2");

  f.calculateEfficiency("pt_eta", "passing_idx", "pass", unbinnedVariables, binnedReals, binnedStates, binToPDFmap, true);

  vector<string> mcBins;
  mcBins.push_back("true");
  binnedStates["mc_true_idx"] = mcBins;

  vector<string> emptyMap; //no fits here

  f.calculateEfficiency("pt_eta_mcTrue", "passing_idx", "pass", unbinnedVariables, binnedReals, binnedStates, emptyMap, true);
}

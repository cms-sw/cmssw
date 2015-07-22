#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/TagAndProbe/interface/TagProbeFitter.h"

using namespace std;
using namespace edm;

class TagProbeFitTreeAnalyzer : public edm::EDAnalyzer{
  public:
    TagProbeFitTreeAnalyzer(const edm::ParameterSet& pset);
    virtual ~TagProbeFitTreeAnalyzer(){};
    virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override {};
    virtual void endRun(const edm::Run &run, const edm::EventSetup &setup) override{};
    void calculateEfficiency(string name, const edm::ParameterSet& pset);
  private:
    TagProbeFitter fitter;
};

TagProbeFitTreeAnalyzer::TagProbeFitTreeAnalyzer(const edm::ParameterSet& pset):
  fitter( pset.getParameter<vector<string> >("InputFileNames"),
          pset.getParameter<string>("InputDirectoryName"),
          pset.getParameter<string>("InputTreeName"),
          pset.getParameter<string>("OutputFileName"),
          pset.existsAs<unsigned int>("NumCPU")?pset.getParameter<unsigned int>("NumCPU"):1,
          pset.existsAs<bool>("SaveWorkspace")?pset.getParameter<bool>("SaveWorkspace"):false,
	  pset.existsAs<bool>("floatShapeParameters")?pset.getParameter<bool>("floatShapeParameters"):true,
	  pset.existsAs<vector<string> >("fixVars")?pset.getParameter<vector<string> >("fixVars"):vector<string>()
	  )
{
  fitter.setQuiet(pset.getUntrackedParameter("Quiet",false));

  if (pset.existsAs<bool>("binnedFit")) {
    bool binned = pset.getParameter<bool>("binnedFit");
    fitter.setBinnedFit(binned, binned ? pset.getParameter<uint32_t>("binsForFit") : 0);
  } else if (pset.existsAs<uint32_t>("binsForMassPlots")) {
    fitter.setBinsForMassPlots(pset.getParameter<uint32_t>("binsForMassPlots"));
  }

  if (pset.existsAs<bool>("saveDistributionsPlot")) {
    fitter.setSaveDistributionsPlot(pset.getParameter<bool>("saveDistributionsPlot"));
  }
  if (pset.existsAs<std::string>("WeightVariable")) {
    fitter.setWeightVar(pset.getParameter<std::string>("WeightVariable"));
  }
  const ParameterSet variables = pset.getParameter<ParameterSet>("Variables");
  vector<string> variableNames = variables.getParameterNamesForType<vector<string> >();
  for (vector<string>::const_iterator name = variableNames.begin(); name != variableNames.end(); name++) {
    vector<string> var = variables.getParameter<vector<string> >(*name);
    double lo, hi;
    if(var.size()>=4 && !(istringstream(var[1])>>lo).fail() && !(istringstream(var[2])>>hi).fail()){
      fitter.addVariable(*name, var[0], lo, hi, var[3]);
    }else{
      LogError("TagProbeFitTreeAnalyzer")<<"Could not create variable: "<<*name<<
      ". Example: pt = cms.vstring(\"Probe pT\", \"1.0\", \"100.0\", \"GeV/c\") ";
    }
  }

  const ParameterSet categories = pset.getParameter<ParameterSet>("Categories");
  vector<string> categoryNames = categories.getParameterNamesForType<vector<string> >();
  for (vector<string>::const_iterator name = categoryNames.begin(); name != categoryNames.end(); name++) {
    vector<string> cat = categories.getParameter<vector<string> >(*name);
    if(cat.size()==2){
      fitter.addCategory(*name, cat[0], cat[1]);
    }else{
      LogError("TagProbeFitTreeAnalyzer")<<"Could not create category: "<<*name<<
      ". Example: mcTrue = cms.vstring(\"MC True\", \"dummy[true=1,false=0]\") ";
    }
  }

  if (pset.existsAs<ParameterSet>("Expressions")) {
    const ParameterSet exprs = pset.getParameter<ParameterSet>("Expressions");
    vector<string> exprNames = exprs.getParameterNamesForType<vector<string> >();
    for (vector<string>::const_iterator name = exprNames.begin(); name != exprNames.end(); name++) {
        vector<string> expr = exprs.getParameter<vector<string> >(*name);
        if(expr.size()>=2){
            vector<string> args(expr.begin()+2,expr.end());
            fitter.addExpression(*name, expr[0], expr[1], args);
        }else{
            LogError("TagProbeFitTreeAnalyzer")<<"Could not create expr: "<<*name<<
                ". Example: qop = cms.vstring(\"qOverP\", \"charge/p\", \"charge\", \"p\") ";
        }
    }
  }


  if (pset.existsAs<ParameterSet>("Cuts")) {
    const ParameterSet cuts = pset.getParameter<ParameterSet>("Cuts");
    vector<string> cutNames = cuts.getParameterNamesForType<vector<string> >();
    for (vector<string>::const_iterator name = cutNames.begin(); name != cutNames.end(); name++) {
        vector<string> cat = cuts.getParameter<vector<string> >(*name);
        if(cat.size()==3){
            fitter.addThresholdCategory(*name, cat[0], cat[1], atof(cat[2].c_str()));
        }else{
            LogError("TagProbeFitTreeAnalyzer")<<"Could not create cut: "<<*name<<
                ". Example: matched = cms.vstring(\"Matched\", \"deltaR\", \"0.5\") ";
        }
    }
  }

  if(pset.existsAs<ParameterSet>("PDFs")){
    const ParameterSet pdfs = pset.getParameter<ParameterSet>("PDFs");
    vector<string> pdfNames = pdfs.getParameterNamesForType<vector<string> >();
    for (vector<string>::const_iterator name = pdfNames.begin(); name != pdfNames.end(); name++) {
      vector<string> pdf = pdfs.getParameter<vector<string> >(*name);
      fitter.addPdf(*name, pdf);
    }
  }

  const ParameterSet efficiencies = pset.getParameter<ParameterSet>("Efficiencies");
  vector<string> efficiencyNames = efficiencies.getParameterNamesForType<ParameterSet>();
  for (vector<string>::const_iterator name = efficiencyNames.begin(); name != efficiencyNames.end(); name++) {
    try {
        calculateEfficiency(*name, efficiencies.getParameter<ParameterSet>(*name));
    } catch (std::exception &ex) {
        throw cms::Exception("Error", ex.what());
    }
  }
}

void TagProbeFitTreeAnalyzer::calculateEfficiency(string name, const edm::ParameterSet& pset){
  vector<string> effCatState = pset.getParameter<vector<string> >("EfficiencyCategoryAndState");
  if(effCatState.empty() ||  (effCatState.size() % 2 == 1)){
    cout<<"EfficiencyCategoryAndState must be a even-sized list of category names and states of that category (cat1, state1, cat2, state2, ...)."<<endl;
    exit(1);
  }

  vector<string> unbinnedVariables;
  if(pset.existsAs<vector<string> >("UnbinnedVariables")){
    unbinnedVariables = pset.getParameter<vector<string> >("UnbinnedVariables");
  }

  const ParameterSet binVars = pset.getParameter<ParameterSet>("BinnedVariables");
  map<string, vector<double> >binnedVariables;
  vector<string> variableNames = binVars.getParameterNamesForType<vector<double> >();
  for (vector<string>::const_iterator var = variableNames.begin(); var != variableNames.end(); var++) {
    vector<double> binning = binVars.getParameter<vector<double> >(*var);
    binnedVariables[*var] = binning;
  }
  map<string, vector<string> >mappedCategories;
  vector<string> categoryNames = binVars.getParameterNamesForType<vector<string> >();
  for (vector<string>::const_iterator var = categoryNames.begin(); var != categoryNames.end(); var++) {
    vector<string> map = binVars.getParameter<vector<string> >(*var);
    mappedCategories[*var] = map;
  }

  vector<string> binToPDFmap;
  if(pset.existsAs<vector<string> >("BinToPDFmap")){
    binToPDFmap = pset.getParameter<vector<string> >("BinToPDFmap");
  }
  if((binToPDFmap.size() > 0) && (binToPDFmap.size()%2 == 0)){
    cout<<"BinToPDFmap must have odd size, first string is the default, followed by binRegExp - PDFname pairs!"<<endl;
    exit(2);
  }

  vector<string> effCats, effStates;
  for (size_t i = 0, n = effCatState.size()/2; i < n; ++i) {
    effCats.push_back(effCatState[2*i]);
    effStates.push_back(effCatState[2*i+1]);
  }

  fitter.calculateEfficiency(name, effCats, effStates, unbinnedVariables, binnedVariables, mappedCategories, binToPDFmap);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TagProbeFitTreeAnalyzer);


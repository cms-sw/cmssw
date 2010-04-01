#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/TagAndProbe/interface/TagProbeFitter.h"

using namespace edm;

class TagProbeFitTreeAnalyzer : public edm::EDAnalyzer{
  public:
    TagProbeFitTreeAnalyzer(const edm::ParameterSet& pset);
    virtual ~TagProbeFitTreeAnalyzer(){};
    virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
    virtual void endRun(const edm::Run &run, const edm::EventSetup &setup){};
    void calculateEfficiency(string name, const edm::ParameterSet& pset);
  private:
    TagProbeFitter fitter;
};

TagProbeFitTreeAnalyzer::TagProbeFitTreeAnalyzer(const edm::ParameterSet& pset):
  fitter( pset.getUntrackedParameter<string>("InputFileName", ""),
          pset.getUntrackedParameter<string>("InputDirectoryName", ""),
          pset.getUntrackedParameter<string>("InputTreeName", ""),
          pset.getUntrackedParameter<string>("OutputFileName", ""),
          pset.getUntrackedParameter<uint>("NumCPU", 1),
          pset.getUntrackedParameter<bool>("SaveWorkspace", false)
  )
{
  const ParameterSet variables = pset.getUntrackedParameter<ParameterSet>("Variables");
  vector<string> variableNames = variables.getParameterNamesForType<vector<string> >(false);
  for (vector<string>::const_iterator name = variableNames.begin(); name != variableNames.end(); name++) {
    vector<string> var = variables.getUntrackedParameter<vector<string> >(*name);
    double lo, hi;
    if(var.size()>=4 && !(istringstream(var[1])>>lo).fail() && !(istringstream(var[2])>>hi).fail()){
      fitter.addVariable(*name, var[0], lo, hi, var[3]);
    }else{
      LogError("TagProbeFitTreeAnalyzer")<<"Could not create variable: "<<*name<<
      ". Example: pt = cms.untracked.vstring(\"Probe pT\", \"1.0\", \"100.0\", \"GeV/c\") ";
    }
  }

  const ParameterSet categories = pset.getUntrackedParameter<ParameterSet>("Categories");
  vector<string> categoryNames = categories.getParameterNamesForType<vector<string> >(false);
  for (vector<string>::const_iterator name = categoryNames.begin(); name != categoryNames.end(); name++) {
    vector<string> cat = categories.getUntrackedParameter<vector<string> >(*name);
    if(cat.size()==2){
      fitter.addCategory(*name, cat[0], cat[1]);
    }else{
      LogError("TagProbeFitTreeAnalyzer")<<"Could not create category: "<<*name<<
      ". Example: mcTrue = cms.untracked.vstring(\"MC True\", \"dummy[true=1,false=0]\") ";
    }
  }

  const ParameterSet pdfs = pset.getUntrackedParameter<ParameterSet>("PDFs");
  vector<string> pdfNames = pdfs.getParameterNamesForType<vector<string> >(false);
  for (vector<string>::const_iterator name = pdfNames.begin(); name != pdfNames.end(); name++) {
    vector<string> pdf = pdfs.getUntrackedParameter<vector<string> >(*name);
    fitter.addPdf(*name, pdf);
  }

  const ParameterSet efficiencies = pset.getUntrackedParameter<ParameterSet>("Efficiencies");
  vector<string> efficiencyNames = efficiencies.getParameterNamesForType<ParameterSet>(false);
  for (vector<string>::const_iterator name = efficiencyNames.begin(); name != efficiencyNames.end(); name++) {
    calculateEfficiency(*name, efficiencies.getUntrackedParameter<ParameterSet>(*name));
  }
}

void TagProbeFitTreeAnalyzer::calculateEfficiency(string name, const edm::ParameterSet& pset){
  vector<string> effCatState = pset.getUntrackedParameter<vector<string> >("EfficiencyCategoryAndState");
  if(effCatState.size() != 2){
    cout<<"EfficiencyCategoryAndState must specify a category and a state of that category"<<endl;
    exit(1);
  }

  vector<string> unbinnedVariables = pset.getUntrackedParameter<vector<string> >("UnbinnedVariables", vector<string>());

  const ParameterSet binVars = pset.getUntrackedParameter<ParameterSet>("BinnedVariables", ParameterSet());
  map<string, vector<double> >binnedVariables;
  vector<string> variableNames = binVars.getParameterNamesForType<vector<double> >(false);
  for (vector<string>::const_iterator var = variableNames.begin(); var != variableNames.end(); var++) {
    vector<double> binning = binVars.getUntrackedParameter<vector<double> >(*var);
    binnedVariables[*var] = binning;
  }
  map<string, vector<string> >mappedCategories;
  vector<string> categoryNames = binVars.getParameterNamesForType<vector<string> >(false);
  for (vector<string>::const_iterator var = categoryNames.begin(); var != categoryNames.end(); var++) {
    vector<string> map = binVars.getUntrackedParameter<vector<string> >(*var);
    mappedCategories[*var] = map;
  }

  vector<string> binToPDFmap = pset.getUntrackedParameter<vector<string> >("BinToPDFmap", vector<string>());
  if((binToPDFmap.size() > 0) && (binToPDFmap.size()%2 == 0)){
    cout<<"BinToPDFmap must have odd size, first string is the default, followed by binRegExp - PDFname pairs!"<<endl;
    exit(2);
  }

  fitter.calculateEfficiency(name, effCatState[0], effCatState[1], unbinnedVariables, binnedVariables, mappedCategories, binToPDFmap, true);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TagProbeFitTreeAnalyzer);


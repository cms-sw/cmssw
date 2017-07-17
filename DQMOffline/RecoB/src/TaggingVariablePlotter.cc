#include "DQMOffline/RecoB/interface/TaggingVariablePlotter.h"

using namespace std;
using namespace edm;
using namespace reco;

TaggingVariablePlotter::VariableConfig::VariableConfig(
        const string &name, const ParameterSet &pSet,
        const string &category, const string& label, unsigned int mc, DQMStore::IBooker & ibook) :
    var(getTaggingVariableName(name)),
    nBins(pSet.getParameter<unsigned int>("nBins")),
    min(pSet.getParameter<double>("min")),
    max(pSet.getParameter<double>("max"))
{
    if (var == btau::lastTaggingVariable)
        throw cms::Exception("Configuration")
            << "Tagging variable \"" << name
            << "\" does not exist." << endl;

    if (pSet.exists("logScale"))
        logScale = pSet.getParameter<bool>("logScale");
    else
        logScale = false;

    std::vector<unsigned int> indices;
    if (pSet.exists("indices"))
        indices = pSet.getParameter< vector<unsigned int> >("indices");
    else
        indices.push_back(0);

    for (const unsigned int iter: indices) {
        Plot plot;
        plot.histo = std::make_shared<FlavourHistograms<double>>(
                                   name + (iter ? Form("%d", iter) : "")
                                   + (category.empty() ? "_" + label
                                  : ("_" + category) + "_" + label),
                                   TaggingVariableDescription[var], nBins, min, max,
                                   false, logScale, true, "b", label,mc,ibook);
        plot.index = iter;
        plots.push_back(plot);
    }
}

TaggingVariablePlotter::TaggingVariablePlotter(const std::string &tagName,
                           const EtaPtBin &etaPtBin, const ParameterSet &pSet,
                           unsigned int mc, bool willFinalize, DQMStore::IBooker & ibook,
                           const string &category): BaseTagInfoPlotter(tagName, etaPtBin), mcPlots_(mc)
{
  const std::string tagVarDir(theExtensionString.substr(1));
  
  if (willFinalize) return;

  const vector<string>& pSets = pSet.getParameterNames();
  for (const std::string& iter: pSets) {
    VariableConfig var(iter,
               pSet.getParameter<ParameterSet>(iter),
               category,tagVarDir, mcPlots_, ibook);
    variables.push_back(var);
  }
}


TaggingVariablePlotter::~TaggingVariablePlotter() {}


void TaggingVariablePlotter::analyzeTag(const BaseTagInfo *baseTagInfo, double jec, int jetFlavour, float w/*=1*/)
{
  analyzeTag(baseTagInfo->taggingVariables(), jetFlavour,w);
}

void TaggingVariablePlotter::analyzeTag(const TaggingVariableList &vars, int jetFlavour, float w/*=1*/)
{
    for (const VariableConfig& cfg: variables) {
        const std::vector<TaggingValue> values(vars.getList(cfg.var, false));
        if (values.empty())
            continue;

        const unsigned int& size = values.size();
        
        for (const VariableConfig::Plot& plot: cfg.plots) {
            if (plot.index == 0) {
                for (const TaggingValue& val: values)
                  plot.histo->fill(jetFlavour, val, w);
            } else if (plot.index - 1 < size)
                plot.histo->fill(jetFlavour, values[plot.index - 1], w);
        }
    }
}

#include "DQMOffline/RecoB/interface/TaggingVariablePlotter.h"

using namespace std;
using namespace edm;
using namespace reco;

TaggingVariablePlotter::VariableConfig::VariableConfig(
		const string &name, const ParameterSet &pSet, const bool& update,
		const string &category, const string& label, const unsigned int& mc) :
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

	for(std::vector<unsigned int>::const_iterator iter = indices.begin();
	    iter != indices.end(); ++iter) {
		Plot plot;
		plot.histo.reset(new FlavourHistograms<double>(
							       name + (*iter ? Form("%d", *iter) : "")
							       + (category.empty() ? "_" + label
								  : ("_" + category) + "_" + label),
			TaggingVariableDescription[var], nBins, min, max,
			false, logScale, true, "b", update,label,mc));
		plot.index = *iter;
		plots.push_back(plot);
	}
}

TaggingVariablePlotter::TaggingVariablePlotter(const std::string &tagName,
					       const EtaPtBin &etaPtBin, const ParameterSet &pSet, const bool& update,
					       const unsigned int& mc,
					       const string &category) : BaseTagInfoPlotter(tagName, etaPtBin), mcPlots_(mc)
{
  const std::string tagVarDir(theExtensionString.substr(1));

	const vector<string>& pSets = pSet.getParameterNames();
	for(vector<string>::const_iterator iter = pSets.begin();
	    iter != pSets.end(); ++iter) {
		VariableConfig var(*iter,
		                   pSet.getParameter<ParameterSet>(*iter),
		                   update, category,tagVarDir, mcPlots_);
		variables.push_back(var);
	}
}


TaggingVariablePlotter::~TaggingVariablePlotter ()
{
}


void TaggingVariablePlotter::analyzeTag (const BaseTagInfo *baseTagInfo,
	const int &jetFlavour)
{
  analyzeTag(baseTagInfo->taggingVariables(), jetFlavour,1.);
}

void TaggingVariablePlotter::analyzeTag (const BaseTagInfo *baseTagInfo,
					 const int &jetFlavour,
					 const float & w)
{
  analyzeTag(baseTagInfo->taggingVariables(), jetFlavour,w);
}

void TaggingVariablePlotter::analyzeTag (const TaggingVariableList &vars,
	const int &jetFlavour)
{
  analyzeTag(vars,jetFlavour,1.);
}

void TaggingVariablePlotter::analyzeTag (const TaggingVariableList &vars,
					 const int &jetFlavour,
					 const float & w)
{
	for(vector<VariableConfig>::const_iterator iter = variables.begin();
	    iter != variables.end(); ++iter) {
		const std::vector<TaggingValue> values(vars.getList(iter->var, false));
		if (values.empty())
			continue;

		const unsigned int& size = values.size();
		for(std::vector<VariableConfig::Plot>::const_iterator plot =
						iter->plots.begin();
		    plot != iter->plots.end(); plot++) {
			if (plot->index == 0) {
				for(std::vector<TaggingValue>::const_iterator iter = values.begin();
                                    iter != values.end(); ++iter)
				  plot->histo->fill(jetFlavour, *iter,w);
			} else if (plot->index - 1 < size)
				plot->histo->fill(jetFlavour,
				                  values[plot->index - 1],w);
		}
	}
}

void TaggingVariablePlotter::finalize()
{
}

void TaggingVariablePlotter::psPlot(const std::string &name)
{
}




void TaggingVariablePlotter::epsPlot(const std::string &name)
{
}

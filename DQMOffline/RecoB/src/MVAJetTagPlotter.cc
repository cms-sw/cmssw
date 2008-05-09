#include <sstream>

#include <boost/bind.hpp>

#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"

#include "DQMOffline/RecoB/interface/MVAJetTagPlotter.h"

using namespace std;
using namespace boost;
using namespace edm;
using namespace reco;

MVAJetTagPlotter::MVAJetTagPlotter(const TString &tagName,
                                   const EtaPtBin &etaPtBin,
                                   const ParameterSet &pSet, bool update) :
	BaseTagInfoPlotter(tagName, etaPtBin),
	jetTagComputer(tagName), computer(0),
	categoryVariable(btau::lastTaggingVariable)
{
	typedef std::vector<ParameterSet> VParameterSet;
	VParameterSet pSets;
	if (pSet.exists("categoryVariable")) {
		categoryVariable = getTaggingVariableName(
			pSet.getParameter<string>("categoryVariable"));
		pSets = pSet.getParameter<VParameterSet>("categories");
	} else
		pSets.push_back(pSet);

	for(unsigned int i = 0; i < pSets.size(); i++) {
		ostringstream ss;
		ss << "CAT" << i;
		categoryPlotters.push_back(
			new TaggingVariablePlotter(tagName, etaPtBin,
			                           pSets[i], update,
			                           i ? ss.str() : string()));
	}
}

MVAJetTagPlotter::~MVAJetTagPlotter ()
{
	for_each(categoryPlotters.begin(), categoryPlotters.end(),
	         bind(&::operator delete, _1));
}

void MVAJetTagPlotter::setEventSetup(const edm::EventSetup &setup)
{
	ESHandle<JetTagComputer> handle;
	setup.get<JetTagComputerRecord>().get(jetTagComputer, handle);
	computer = dynamic_cast<const GenericMVAJetTagComputer*>(handle.product());

	if (!computer)
		throw cms::Exception("Configuration")
			<< "JetTagComputer passed to "
			   "MVAJetTagPlotter::analyzeTag is not a "
			   "GenericMVAJetTagComputer." << endl;
}

void MVAJetTagPlotter::analyzeTag (const vector<const BaseTagInfo*> &baseTagInfos,
                                   const int &jetFlavour)
{
	// taggingVariables() should not need EventSetup
	// computer->setEventSetup(es);

	JetTagComputer::TagInfoHelper helper(baseTagInfos);
	TaggingVariableList vars = computer->taggingVariables(helper);

	categoryPlotters[0]->analyzeTag(vars, jetFlavour);
	if (categoryVariable != btau::lastTaggingVariable) {
		
		unsigned int cat;
		try {
			cat = (unsigned int)vars.get(categoryVariable) + 1;
		} catch(edm::Exception e) {
			// no category for this jet tag
			// this means the jet didn't pass any cuts, ignore
			return;
		}
		if (cat >= 1 && cat < categoryPlotters.size())
			categoryPlotters[cat]->analyzeTag(vars, jetFlavour);
	}
}

void MVAJetTagPlotter::finalize()
{
	for_each(categoryPlotters.begin(), categoryPlotters.end(),
	         bind(&TaggingVariablePlotter::finalize, _1));
}

void MVAJetTagPlotter::psPlot(const TString &name)
{
	for_each(categoryPlotters.begin(), categoryPlotters.end(),
	         bind(&TaggingVariablePlotter::psPlot, _1, ref(name)));
}

/*void MVAJetTagPlotter::write(const bool allHisto)
{
	for_each(categoryPlotters.begin(), categoryPlotters.end(),
	         bind(&TaggingVariablePlotter::write, _1, allHisto));
}*/

void MVAJetTagPlotter::epsPlot(const TString &name)
{
	for_each(categoryPlotters.begin(), categoryPlotters.end(),
	         bind(&TaggingVariablePlotter::epsPlot, _1, ref(name)));
}

vector<string> MVAJetTagPlotter::tagInfoRequirements() const
{
	return computer->getInputLabels();
}

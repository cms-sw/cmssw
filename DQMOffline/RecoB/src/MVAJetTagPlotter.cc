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

MVAJetTagPlotter::MVAJetTagPlotter(const std::string &tagName,
                                   const EtaPtBin &etaPtBin,
                                   const ParameterSet &pSet,
                                   const std::string& folderName,
                                   unsigned int mc, bool willFinalize, 
				   DQMStore::IBooker & ibook):
	BaseTagInfoPlotter(folderName, etaPtBin),
	jetTagComputer(tagName), computer(0),
	categoryVariable(btau::lastTaggingVariable)
{
	typedef std::vector<ParameterSet> VParameterSet;
	VParameterSet pSets;
	if (pSet.exists("categoryVariable")) {
		categoryVariable = getTaggingVariableName(
			pSet.getParameter<string>("categoryVariable"));
		pSets = pSet.getParameter<VParameterSet>("categories");
	} else pSets.push_back(pSet);

	for (unsigned int i = 0; i != pSets.size(); ++i) {
      std::string ss = "CAT";
      ss += std::to_string(i);
	  categoryPlotters.push_back(
              std::make_unique<TaggingVariablePlotter>(folderName, etaPtBin,
								pSets[i], mc, willFinalize, ibook,
								i ? ss : std::string())
                     );
	}
}

MVAJetTagPlotter::~MVAJetTagPlotter() { }

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

void MVAJetTagPlotter::analyzeTag(const vector<const BaseTagInfo*> &baseTagInfos, double jec, int jetFlavour, float w/*=1*/)
{
	const JetTagComputer::TagInfoHelper helper(baseTagInfos);
	const TaggingVariableList& vars = computer->taggingVariables(helper);

	categoryPlotters.front()->analyzeTag(vars, jetFlavour, w);
	if (categoryVariable != btau::lastTaggingVariable) {
		unsigned int cat = (unsigned int)(vars.get(categoryVariable, -1) + 1);
		if (cat >= 1 && cat < categoryPlotters.size())
		  categoryPlotters[cat]->analyzeTag(vars, jetFlavour, w);
	}
}

void MVAJetTagPlotter::finalize(DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_)
{
  //nothing done here in principle and function below does not work
  /*
	for_each(categoryPlotters.begin(), categoryPlotters.end(),
		 boost::bind(&TaggingVariablePlotter::finalize, _1, boost::ref(ibook_), _2, boost::ref(igetter_)));
  */
}

void MVAJetTagPlotter::psPlot(const std::string &name)
{
    for (const auto& plot: categoryPlotters)
        plot->psPlot(name);
}

void MVAJetTagPlotter::epsPlot(const std::string &name)
{
    for (const auto& plot: categoryPlotters)
        plot->epsPlot(name);
}

vector<string> MVAJetTagPlotter::tagInfoRequirements() const
{
	vector<string> labels = computer->getInputLabels();
	if (labels.empty())
		labels.push_back("tagInfos");
	return labels;
}

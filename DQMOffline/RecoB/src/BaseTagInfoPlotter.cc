#include <vector>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"

using namespace std;
using namespace reco;

void BaseTagInfoPlotter::analyzeTag(const BaseTagInfo * tagInfo, double jec, int jetFlavour, float w/*=1*/)
{
  throw cms::Exception("MissingVirtualMethod")
  	<< "No analyzeTag method overloaded from BaseTagInfoPlotter." << endl;
}

void BaseTagInfoPlotter::analyzeTag(const vector<const BaseTagInfo *> &tagInfos, double jec, int jetFlavour, float w/*=1*/)
{

  if (tagInfos.size() != 1)
    throw cms::Exception("MismatchedTagInfos")
    	<< tagInfos.size() << " BaseTagInfos passed, but only one expected." << endl;

  analyzeTag(tagInfos.front(), jec, jetFlavour, w);

}

void BaseTagInfoPlotter::setEventSetup(const edm::EventSetup & setup)
{
}

vector<string> BaseTagInfoPlotter::tagInfoRequirements() const
{
  return vector<string>();
}

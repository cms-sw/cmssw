#include <vector>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"

using namespace std;
using namespace reco;

void BaseTagInfoPlotter::analyzeTag(const BaseTagInfo * tagInfo, const int & jetFlavour)
{
  throw cms::Exception("MissingVirtualMethod")
  	<< "No analyzeTag method overloaded from BaseTagInfoPlotter." << endl;
}

void BaseTagInfoPlotter::analyzeTag(const BaseTagInfo * tagInfo, const int & jetFlavour, const float & w)
{
  throw cms::Exception("MissingVirtualMethod")
  	<< "No analyzeTag method overloaded from BaseTagInfoPlotter." << endl;
}

void BaseTagInfoPlotter::analyzeTag(const vector<const BaseTagInfo *> &tagInfos, const int & jetFlavour, const float & w)
{

  if (tagInfos.size() != 1)
    throw cms::Exception("MismatchedTagInfos")
    	<< tagInfos.size() << " BaseTagInfos passed, but only one expected." << endl;

  analyzeTag(tagInfos.front(), jetFlavour, w);

}

void BaseTagInfoPlotter::analyzeTag(const vector<const BaseTagInfo *> &tagInfos, const int & jetFlavour)
{
  if (tagInfos.size() != 1)
    throw cms::Exception("MismatchedTagInfos")
      << tagInfos.size() << " BaseTagInfos passed, but only one expected." << endl;
  
  analyzeTag(tagInfos.front(), jetFlavour);
}

void BaseTagInfoPlotter::setEventSetup(const edm::EventSetup & setup)
{
}

vector<string> BaseTagInfoPlotter::tagInfoRequirements() const
{
  return vector<string>();
}

#include "DQMOffline/Lumi/interface/RunLumiRangeMap.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/lexical_cast.hpp>

using namespace ZCountingTrigger;

//--------------------------------------------------------------------------------------------------
void RunLumiRangeMap::addJSONFile(const std::string &filepath)
{
  // read json file into boost property tree
  boost::property_tree::ptree jsonTree;
  boost::property_tree::read_json(filepath,jsonTree);
  
  // loop through boost property tree and fill the MapType structure with the list of good lumi ranges for each run
  for(boost::property_tree::ptree::const_iterator it = jsonTree.begin(); it!=jsonTree.end(); ++it) {
    unsigned int runNum = boost::lexical_cast<unsigned int>(it->first);
    MapType::mapped_type &lumiPairList = fMap[runNum];
    boost::property_tree::ptree lumiPairListTree = it->second;
    for(boost::property_tree::ptree::const_iterator jt = lumiPairListTree.begin(); jt!=lumiPairListTree.end(); ++jt) {
      boost::property_tree::ptree lumiPairTree = jt->second;
      if(lumiPairTree.size()==2) {
        unsigned int firstLumi = boost::lexical_cast<unsigned int>(lumiPairTree.begin()->second.data());
        unsigned int lastLumi  = boost::lexical_cast<unsigned int>((++lumiPairTree.begin())->second.data());
        lumiPairList.push_back(std::pair<unsigned int, unsigned int>(firstLumi,lastLumi));
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
bool RunLumiRangeMap::hasRunLumi(const RunLumiPairType &runLumi) const
{
  // Check if a given run,lumi pair is included in the mapped lumi ranges

  // check if run is included in the map
  MapType::const_iterator it = fMap.find(runLumi.first);
  if(it!=fMap.end()) {
    //check lumis
    const MapType::mapped_type &lumiPairList = it->second;
    for(MapType::mapped_type::const_iterator jt = lumiPairList.begin(); jt<lumiPairList.end(); ++jt) {
      if(runLumi.second >= jt->first && runLumi.second <= jt->second) {
        //found lumi in accepted range
        return true;
      }
    }
  }

  return false;
}

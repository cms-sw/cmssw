#ifndef DQMOFFLINE_LUMIZCOUNTING_RUNLUMIRANGEMAP_H
#define DQMOFFLINE_LUMIZCOUNTING_RUNLUMIRANGEMAP_H

#include <string>
#include <vector>
#include <map>

namespace baconhep
{
  class RunLumiRangeMap
  {
    public:
      typedef std::pair<unsigned int, unsigned int>                RunLumiPairType;
      typedef std::map<unsigned int, std::vector<RunLumiPairType>> MapType;
      
      RunLumiRangeMap(){}
      ~RunLumiRangeMap(){}
      
      void addJSONFile(const std::string &filepath);
      bool hasRunLumi(const RunLumiPairType &runLumi) const;
    
    protected:
      MapType fMap; // mapped run-lumi ranges to accept
  };
}
#endif

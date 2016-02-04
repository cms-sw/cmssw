#ifndef FWCore_ParameterSet_ParameterSetConverter_h
#define FWCore_ParameterSet_ParameterSetConverter_h

#include <map>
#include <list>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "boost/utility.hpp"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  typedef std::vector<std::string> StringVector;

  struct MainParameterSet {
    MainParameterSet(ParameterSetID const& oldID, std::string const& psetString);
    ~MainParameterSet();
    ParameterSetID oldID_;
    ParameterSet parameterSet_;
    StringVector paths_;
    StringVector endPaths_;
    std::set<std::string> triggerPaths_;
  };

  struct TriggerPath {
    TriggerPath(ParameterSet const& pset);
    ~TriggerPath();
    ParameterSet parameterSet_;
    StringVector tPaths_;
    std::set<std::string> triggerPaths_;
  };

  //------------------------------------------------------------
  // Class ParameterSetConverter

  typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
  class ParameterSetConverter : private boost::noncopyable {
  public:
    typedef std::list<std::string> StringList;
    typedef std::map<std::string, std::string> StringMap;
    typedef std::list<std::pair<std::string, ParameterSetID> > StringWithIDList;
    typedef std::map<ParameterSetID, ParameterSetID> ParameterSetIdConverter;
    ParameterSetConverter(ParameterSetMap const& psetMap, ParameterSetIdConverter& idConverter, bool alreadyByReference);
    ~ParameterSetConverter();
    ParameterSetIdConverter const& parameterSetIdConverter() const {return parameterSetIdConverter_;}
  private:
    void convertParameterSets();
    void noConvertParameterSets();
    StringWithIDList parameterSets_;
    std::vector<MainParameterSet> mainParameterSets_;
    std::vector<TriggerPath> triggerPaths_;
    StringMap replace_;
    ParameterSetIdConverter& parameterSetIdConverter_;
  };
}
#endif

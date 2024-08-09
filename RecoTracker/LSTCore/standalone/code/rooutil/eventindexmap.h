#ifndef eventindexmap_h
#define eventindexmap_h

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>

#include "TString.h"
#include "TEventList.h"

#include "printutil.h"

namespace RooUtil {
  class EventIndexMap {
  public:
    std::map<TString, TEventList*> eventlistmap_;
    EventIndexMap();
    ~EventIndexMap();
    void load(TString filename);
    bool hasEventList(TString);
    TEventList* getEventList(TString);
  };
}  // namespace RooUtil

#endif

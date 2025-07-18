#include "eventindexmap.h"

using namespace RooUtil;

RooUtil::EventIndexMap::EventIndexMap() {}
RooUtil::EventIndexMap::~EventIndexMap() {}

//_____________________________________________________________________________________
void RooUtil::EventIndexMap::load(TString filename) {
  eventlistmap_.clear();

  std::ifstream ifile;
  ifile.open(filename.Data());
  std::string line;

  while (std::getline(ifile, line)) {
    std::string cms4path;
    int number_of_events;
    TEventList* event_indexs = new TEventList(cms4path.c_str());
    unsigned int event_index;

    std::stringstream ss(line);

    ss >> cms4path >> number_of_events;

    for (int ii = 0; ii < number_of_events; ++ii) {
      ss >> event_index;
      event_indexs->Enter(event_index);
    }

    eventlistmap_[cms4path] = event_indexs;
  }
}

//_____________________________________________________________________________________
bool RooUtil::EventIndexMap::hasEventList(TString cms4file) {
  return eventlistmap_.find(cms4file) != eventlistmap_.end();
}

//_____________________________________________________________________________________
TEventList* RooUtil::EventIndexMap::getEventList(TString cms4file) {
  if (not hasEventList(cms4file))
    error(TString::Format("Does not have the event list for the input %s but asked for it!", cms4file.Data()),
          __FUNCTION__);

  return eventlistmap_[cms4file];
}

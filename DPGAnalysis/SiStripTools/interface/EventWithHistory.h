#ifndef DPGAnalysis_SiStripTools_EventWithHistory_H
#define DPGAnalysis_SiStripTools_EventWithHistory_H

#include <vector>
#include "DPGAnalysis/SiStripTools/interface/TinyEvent.h"

namespace edm{
  class Event;
  class EventAuxiliary;
}

class EventBXHistory;
class L1AcceptBunchCrossing;
typedef std::vector<L1AcceptBunchCrossing> L1AcceptBunchCrossingCollection;

class EventWithHistory: public TinyEvent {

 public:
  
  EventWithHistory();
  explicit EventWithHistory(const TinyEvent& se);
  explicit EventWithHistory(const edm::EventNumber_t event,const int orbit,const int bx);
  explicit EventWithHistory(const edm::EventNumber_t event,const unsigned int orbit,const int bx);
  explicit EventWithHistory(const edm::Event& event);
  EventWithHistory(const std::vector<edm::EventAuxiliary>& he);
  EventWithHistory(const edm::Event& event, const L1AcceptBunchCrossingCollection& l11bcc, 
		   const long long orbitoffset=0, const int bxoffset=0);
  EventWithHistory(const EventWithHistory& he);

  EventWithHistory& operator=(const EventWithHistory& he);

  //  int operator<(const EventWithHistory& other) const;
  int operator==(const EventWithHistory& other) const;

  int add(const EventWithHistory& he, const int idepth);  // return false if depth limit reached or not consecutive
  int add(const TinyEvent& he, const int idepth);  // return false if depth limit reached or not consecutive

  const edm::EventNumber_t event() const;
  const unsigned int orbit() const;
  const int bx() const;

  const TinyEvent* get(const unsigned int ev) const;
  unsigned int depth() const;
  bool isFutureHistory() const;

  long long deltaBX(const unsigned int ev2, const unsigned int ev1) const;
  long long deltaBX(const unsigned int ev1) const;
  long long deltaBX() const;
  long long deltaBX(const TinyEvent& se) const;

  long long absoluteBX(const unsigned int ev1) const;
  long long absoluteBX() const;

  long long absoluteBXinCycle(const unsigned int ev1, const int bx0) const;
  long long absoluteBXinCycle(const int bx0) const;

  long long deltaBXinCycle(const unsigned int ev2, const unsigned int ev1, const int bx0) const;
  long long deltaBXinCycle(const unsigned int ev1, const int bx0) const;
  long long deltaBXinCycle(const int bx0) const;
  long long deltaBXinCycle(const TinyEvent& se, const int bx0) const;

 private:

  std::vector<TinyEvent> _prevse;

};




#endif //  DPGAnalysis_SiStripTools_EventWithHistory_H


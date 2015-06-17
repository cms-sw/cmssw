#include <map>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"

EventWithHistory::EventWithHistory(): TinyEvent(), _prevse() { }

EventWithHistory::EventWithHistory(const TinyEvent& se): TinyEvent(se), _prevse() { }

EventWithHistory::EventWithHistory(const edm::EventNumber_t event, const int orbit, const int bx): 
  TinyEvent(event,orbit,bx), _prevse() { }

EventWithHistory::EventWithHistory(const edm::EventNumber_t event, const unsigned int orbit, const int bx): 
  TinyEvent(event,orbit,bx), _prevse() { }

EventWithHistory::EventWithHistory(const edm::Event& event): 
  TinyEvent(event), _prevse() { }

EventWithHistory::EventWithHistory(const std::vector<edm::EventAuxiliary>& veaux):
  TinyEvent((veaux.size()>0) ? veaux[veaux.size()-1]: TinyEvent()), _prevse() 
{
  for(std::vector<edm::EventAuxiliary>::const_reverse_iterator eaux=veaux.rbegin();eaux!=veaux.rend();eaux++) {
    if(eaux!=veaux.rbegin()) {
      _prevse.push_back(*eaux);
    }
  }
}

EventWithHistory::EventWithHistory(const edm::Event& event, const L1AcceptBunchCrossingCollection& l1abcc,
				   const long long orbitoffset, const int bxoffset):
  TinyEvent(), _prevse()
{
  
  std::map<int,TinyEvent> tmpmap;
  
  for(L1AcceptBunchCrossingCollection::const_iterator l1abc=l1abcc.begin();l1abc!=l1abcc.end();++l1abc) {
    if(event.id().event() > (edm::EventNumber_t)(-1*l1abc->l1AcceptOffset()) ) {
      edm::EventNumber_t evnumb = event.id().event()+l1abc->l1AcceptOffset();
      if(orbitoffset < (long long)l1abc->orbitNumber()) {
	unsigned int neworbit = l1abc->orbitNumber() - orbitoffset;
	int newbx = l1abc->bunchCrossing() - bxoffset;
	
	while(newbx > 3563) {
	  ++neworbit;
	  newbx -= 3564;
	}
	while(newbx < 0) {
	  --neworbit;
	  newbx += 3564;
	}
	
	if(l1abc->eventType()!=0) {
	  TinyEvent tmpse(evnumb,neworbit,newbx);
	  tmpmap[l1abc->l1AcceptOffset()]=tmpse;
	}
	else {
	  edm::LogWarning("L1AcceptBunchCrossingNoType") << "L1AcceptBunchCrossing with no type found: ";
	  for(L1AcceptBunchCrossingCollection::const_iterator debu=l1abcc.begin();debu!=l1abcc.end();++debu) {
	    edm::LogPrint("L1AcceptBunchCrossingNoType") << *debu;
	  }
	}
      }
      else {
	edm::LogError("L1AcceptBunchCrossingOffsetTooLarge") << " Too large orbit offset "
							     << orbitoffset << " " 
							     << l1abc->orbitNumber();
      }
    }
    else {
      edm::LogInfo("L1AcceptBunchCrossingNegativeEvent") << "L1AcceptBunchCrossing with negative event: ";
      for(L1AcceptBunchCrossingCollection::const_iterator debu=l1abcc.begin();debu!=l1abcc.end();++debu) {
	edm::LogVerbatim("L1AcceptBunchCrossingNegativeEvent") << *debu;
      }
    }
  }
  // look for the event itself
  if(tmpmap.find(0)!=tmpmap.end()) {
    
    TinyEvent::operator=(tmpmap[0]);
  
    // loop on the rest of the map and stop when it is missing
    // check that the events are in the right order and break if not

    int counter=-1;
    while(tmpmap.find(counter)!=tmpmap.end()) {

      if(tmpmap[counter+1].deltaBX(tmpmap[counter]) > 0) {
	_prevse.push_back(tmpmap[counter]);
	--counter;
      }
      else {
	edm::LogWarning("L1AcceptBunchCrossingNotInOrder") << "L1AcceptBunchCrossing not in order: orbit " 
							   << event.orbitNumber() << " BX " << event.bunchCrossing()
							   << " orbit offset " << orbitoffset << " bx offset " << bxoffset << " :";
	for(L1AcceptBunchCrossingCollection::const_iterator debu=l1abcc.begin();debu!=l1abcc.end();++debu) {
	  edm::LogPrint("L1AcceptBunchCrossingNotInOrder") << *debu;
	}
	break;
      }
    }
  }
  else {
    TinyEvent::operator=(event);
    edm::LogWarning("L1AcceptBunchCrossingNoCollection") << " L1AcceptBunchCrossing with offset=0 not found "
							 << " likely L1ABCCollection is empty ";
  }
  
}

EventWithHistory::EventWithHistory(const EventWithHistory& he): TinyEvent(he), _prevse(he._prevse) { }

EventWithHistory& EventWithHistory::operator=(const EventWithHistory& he) {

  if(this != &he) {
    TinyEvent::operator=(he);
    _prevse = he._prevse;
  }
  return *this;
}

// int EventWithHistory::operator<(const EventWithHistory& other) const { return TinyEvent::operator<(other); }

int EventWithHistory::operator==(const EventWithHistory& other) const {

  int equal = TinyEvent::operator==(other);

  // depth is not checked anymore

  //  equal = equal && (depth() == other.depth());

  if(equal) {
    for(unsigned int i=0;i<((depth()<other.depth())?depth():other.depth());i++) {
      equal = equal && (_prevse[i] == other._prevse[i]);
    }
  }

  return equal;
}

int EventWithHistory::add(const EventWithHistory& he, const int idepth) {

  if(!add((const TinyEvent&) he,idepth)) return 0;

  for(std::vector<TinyEvent>::const_iterator ev=he._prevse.begin();ev!=he._prevse.end();ev++) {
    if(!add(*ev,idepth)) return 0;
  }
  return 1;
}

int EventWithHistory::add(const TinyEvent& se, const int idepth) {

  bool isfuture = (idepth <0);
  const unsigned int adepth = abs(idepth);

  // protect against the possibility of filling with past and future history

  if( depth()>0 && ((isfuture && !isFutureHistory()) || (!isfuture && isFutureHistory()))) return 0; 

  if(adepth==0) return 0;
  if(_prevse.size()>= adepth) return 0;

  if(_prevse.size()==0) {
    if((!isfuture && isNextOf(se)) || (isfuture && se.isNextOf(*this))) {
      _prevse.push_back(se);
      return 1;
    }
    else {
      return 0;
    }
  }
  else {
    if((!isfuture && _prevse[_prevse.size()-1].isNextOf(se)) || (isfuture && se.isNextOf(_prevse[_prevse.size()-1]))) {
      _prevse.push_back(se);
      return 1;
    }
    else {
      return 0;
    }
  }    
  return 0;
}

const edm::EventNumber_t EventWithHistory::event() const { return TinyEvent::_event; } 
const unsigned int EventWithHistory::orbit() const { return TinyEvent::_orbit; } 
const int EventWithHistory::bx() const { return TinyEvent::_bx; } 

const TinyEvent* EventWithHistory::get(const unsigned int ev) const {

  if(ev==0) return this;
  if(ev<=_prevse.size()) return &_prevse[ev-1];
  return 0;

}

unsigned int EventWithHistory::depth() const { return _prevse.size(); }

bool EventWithHistory::isFutureHistory() const { 

  return (depth()>0 && _prevse[0].isNextOf(*this));

}
 
long long EventWithHistory::deltaBX(const unsigned int ev2, const unsigned int ev1) const {

  if(ev2==ev1) return 0;

  if(ev2<ev1 && ev1<=_prevse.size()) {
    if(ev2==0) return  TinyEvent::deltaBX(_prevse[ev1-1]);
    return _prevse[ev2-1].deltaBX(_prevse[ev1-1]);
  }

  return -1;
}

long long EventWithHistory::deltaBX(const unsigned int ev1) const {  return deltaBX(0,ev1); }

long long EventWithHistory::deltaBX() const {  return deltaBX(0,1); }

long long EventWithHistory::deltaBX(const TinyEvent& se) const { 

 return TinyEvent::deltaBX(se); 

}

long long EventWithHistory::absoluteBX(const unsigned int ev1) const {

  if(ev1==0) return TinyEvent::absoluteBX();
  if(ev1<=_prevse.size()) return _prevse[ev1-1].absoluteBX();

  return -1;

}

long long EventWithHistory::absoluteBX() const {

  return TinyEvent::absoluteBX();

}

long long EventWithHistory::absoluteBXinCycle(const unsigned int ev1, const int bx0) const {

  if(ev1==0) return TinyEvent::absoluteBXinCycle(bx0);
  if(ev1<=_prevse.size()) return _prevse[ev1-1].absoluteBXinCycle(bx0);

  return -1;

}

long long EventWithHistory::absoluteBXinCycle(const int bx0) const {

  return TinyEvent::absoluteBXinCycle(bx0);

}

long long EventWithHistory::deltaBXinCycle(const unsigned int ev2, const unsigned int ev1, const int bx0) const {

  if(ev2==ev1 && ev1<=_prevse.size()) {
    if(ev2==0) return TinyEvent::deltaBXinCycle(*this,bx0);
    return _prevse[ev2-1].deltaBXinCycle(_prevse[ev1-1],bx0);
  }

  if(ev2<ev1 && ev1<=_prevse.size()) {
    if(ev2==0) return  TinyEvent::deltaBXinCycle(_prevse[ev1-1],bx0);
    return _prevse[ev2-1].deltaBXinCycle(_prevse[ev1-1],bx0);
  }

  return -1;
}

long long EventWithHistory::deltaBXinCycle(const unsigned int ev1, const int bx0) const {
  return deltaBXinCycle(0,ev1,bx0);
}

long long EventWithHistory::deltaBXinCycle(const int bx0) const {
  return deltaBXinCycle(0,1,bx0);
}

long long EventWithHistory::deltaBXinCycle(const TinyEvent& se, const int bx0) const {

  return TinyEvent::deltaBXinCycle(se,bx0);

}

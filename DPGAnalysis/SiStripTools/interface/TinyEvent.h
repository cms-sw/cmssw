#ifndef DPGanalysis_SiStripTools_TinyEvent_H
#define DPGanalysis_SiStripTools_TinyEvent_H

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include <vector>

namespace edm {
  class Event;
}

//class EventIdWithBX;

struct TinyEvent {

  TinyEvent():_event(0), _orbit(0), _bx(0) { }

  explicit TinyEvent(const edm::EventNumber_t event,const int orbit,const int bx):
    _event(event), _orbit(orbit<0 ? 0 : orbit ), _bx(bx<0 ? 0 : bx) { }

  explicit TinyEvent(const edm::EventNumber_t event,const unsigned int orbit,const int bx):
    _event(event), _orbit(orbit), _bx(bx<0 ? 0 : bx) { }

  TinyEvent(const TinyEvent& se):
    _event(se._event), _orbit(se._orbit), _bx(se._bx) { }

  TinyEvent(const edm::Event& event);
  TinyEvent(const edm::EventAuxiliary& eaux);

  TinyEvent& operator=(const TinyEvent& se) {

    if(this != &se) {
      _event = se._event;
      _orbit = se._orbit;
      _bx = se._bx;
    }
    
    return *this;

  }

  int operator<(const TinyEvent& other) const { 
    return _event < other._event;
  }

  int operator==(const TinyEvent& other) const { 
    return (_event == other._event) &&
      (_orbit == other._orbit) &&
      (_bx == other._bx) ;
  }

  int isNextOf(const TinyEvent& se) const {

    return (se._event>0) && ((se._event + 1) == _event);

  }

  long long absoluteBX() const {

    return (long long)_orbit * 3564 + _bx;

  }

  long long absoluteBXinCycle(const int bx0) const {

    return (absoluteBX()-bx0);

  }

  long long deltaBX(const TinyEvent& se) const {

    int sign = 1;
    if(se._event > _event) sign = -1;
      
    long long dorb = (int)(_orbit) - (int)(se._orbit);
    return (dorb*3564 + (int)_bx - (int)se._bx) * sign;

  }

  long long deltaBXinCycle(const TinyEvent& se, const int bx0) const {

    long long dbx = deltaBX(se);

    return dbx+(69 + se.absoluteBX() -bx0%70 )%70; // I could use: dbx+(69+se.absoluteBXinCycle(bx0))%70

  }

  edm::EventNumber_t _event;
  unsigned int _orbit;
  unsigned int _bx;

};

typedef std::vector<TinyEvent> TinyEventCollection;

#endif // DPGanalysis_SiStripTools_TinyEvent_H

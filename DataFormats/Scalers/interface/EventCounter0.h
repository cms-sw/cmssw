/*
 *  File: DataFormats/Scalers/interface/EventCounter0.h   (W.Badgett)
 *
 *  EventCounter0 orbit marker information for the EC0 signals prior 
 *  to the current L1 accept.  Currently only the most recent EC0 orbit 
 *  number marker is kept.
 *
 *  orbitNumber_     32-bits     Orbit counter since last reset
 *  offset_                      Offset of the EventCounter0 relative to 
 *                               the previous EventCounter0 where 0 
 *                               means the most recent EC0, -1 next most recent
 *  spare[]        5x64 bits     for future use
 */

#ifndef DATAFORMATS_SCALERS_EVENTCOUNTER0_H
#define DATAFORMATS_SCALERS_EVENTCOUNTER0_H

#include <ostream>
#include <vector>

/*! \file EventCounter0s.h
 * \Header file for EventCounter0 orbit marker the scalers system
 * 
 * \author: William Badgett
 *
 */


/// \class EventCounter0s.h
/// \brief Persistable copy of Scalers EventCounter0 orbit marker

class EventCounter0
{
 public:

  EventCounter0();
  EventCounter0(const int      offset__,
		const unsigned int orbitNumber__,
		const unsigned long long * spare__);
  EventCounter0(const int index,
		const unsigned long long * data);
  virtual ~EventCounter0();

  enum
  {
    ORBIT_NUMBER_SHIFT   = 32ULL,
    ORBIT_NUMBER_MASK    = 0xFFFFFFFFULL,
    N_SPARE              = 5
  };

  /// name method
  std::string name() const { return "EventCounter0"; }

  /// empty method (= false)
  bool empty() const { return false; }

  int offset() const                            { return(offset_);}
  unsigned int orbitNumber() const              { return(orbitNumber_);}
  unsigned long long spare(int i) const         { return(spare_[i]);}
  std::vector<unsigned long long> spare() const { return(spare_);}

  /// equality operator
  int operator==(const EventCounter0& e) const { return false; }

  /// inequality operator
  int operator!=(const EventCounter0& e) const { return false; }

protected:

  int                             offset_;
  unsigned int                    orbitNumber_;
  std::vector<unsigned long long> spare_;
};

/// Pretty-print operator for EventCounter0s
std::ostream& operator<<(std::ostream& s, const EventCounter0& c);

typedef std::vector<EventCounter0> EventCounter0Collection;

#endif

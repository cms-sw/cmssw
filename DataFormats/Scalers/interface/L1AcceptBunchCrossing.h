/*
 *  File: DataFormats/Scalers/interface/L1AcceptBunchCrossing.h   (W.Badgett)
 *
 *  Bunch crossing information for a single L1 accept.  The information 
 *  comes from the Scalers FED.   Normally there are four of these objects 
 *  per event, each representing the current L1 accept, as well as the 
 *  previous three L1 accepts in time.
 *
 *  orbitNumber_     32-bits     Orbit counter since last reset
 *  bunchCrossing_   0...3563    Bunch counter within orbit
 *  eventType_       0...7       Event Type: 
 *                               1: Physics
 *                               2: Calibration
 *                               3: Test
 *                               4: Technical
 *  l1AcceptOffset_              Offset of the L1 Accept relative to 
 *                               the current event's L1 accept
 *                               Typically 0, -1, -2, -3
 */

#ifndef DATAFORMATS_SCALERS_L1ACCEPTBUNCHCROSSING_H
#define DATAFORMATS_SCALERS_L1ACCEPTBUNCHCROSSING_H

#include <ostream>
#include <vector>

/*! \file L1AcceptBunchCrossings.h
 * \Header file for L1Accept bunch crossing data from the scalers system
 * 
 * \author: William Badgett
 *
 */


/// \class L1AcceptBunchCrossings.h
/// \brief Persistable copy of Scalers L1Accept bunch crossing info

class L1AcceptBunchCrossing
{
 public:

  L1AcceptBunchCrossing();
  L1AcceptBunchCrossing(const int l1AcceptOffset__,
			const unsigned int orbitNumber__,
			const unsigned int bunchCrossing__,
			const unsigned int eventType__);
  L1AcceptBunchCrossing(const int index,
			const unsigned long long data);
  virtual ~L1AcceptBunchCrossing();

  enum
  {
    ORBIT_NUMBER_SHIFT   = 32ULL,
    ORBIT_NUMBER_MASK    = 0xFFFFFFFFULL,
    BUNCH_CROSSING_SHIFT = 4ULL,
    BUNCH_CROSSING_MASK  = 0xFFFULL,
    EVENT_TYPE_SHIFT     = 0,
    EVENT_TYPE_MASK      = 0xFULL
  };

  /// name method
  std::string name() const { return "L1AcceptBunchCrossing"; }

  /// empty method (= false)
  bool empty() const { return false; }

  int l1AcceptOffset() const         { return(l1AcceptOffset_);}
  unsigned int orbitNumber() const   { return(orbitNumber_);}
  unsigned int bunchCrossing() const { return(bunchCrossing_);}
  unsigned int eventType() const     { return(eventType_);}

  /// equality operator
  int operator==(const L1AcceptBunchCrossing& e) const { return false; }

  /// inequality operator
  int operator!=(const L1AcceptBunchCrossing& e) const { return false; }

protected:

  int          l1AcceptOffset_;
  unsigned int orbitNumber_;
  unsigned int bunchCrossing_;
  unsigned int eventType_;

};

/// Pretty-print operator for L1AcceptBunchCrossings
std::ostream& operator<<(std::ostream& s, const L1AcceptBunchCrossing& c);

typedef std::vector<L1AcceptBunchCrossing> L1AcceptBunchCrossingCollection;

#endif

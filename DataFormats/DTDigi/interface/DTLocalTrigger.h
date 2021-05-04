#ifndef DTLocalTrigger_DTLocalTrigger_h
#define DTLocalTrigger_DTLocalTrigger_h

/** \class DTLocalTrigger
 *
 * Trigger from DT chamber
 *  
 *
 * \author FRC
 *
 */

#include <cstdint>

class DTLocalTrigger {
public:
  /// Constructor
  explicit DTLocalTrigger(int eventBx, int bx, int data);

  /// Default construction.
  DTLocalTrigger();

  /// triggers are equal if they are in the same chamber and have same BX count (??)
  bool operator==(const DTLocalTrigger& trig) const;

  uint16_t eventBx() const;
  uint16_t bx() const;
  uint16_t quality() const;
  uint16_t trTheta() const;
  bool secondTrack() const;
  bool trOut() const;

  /// Print content of trigger
  void print() const;

private:
  uint16_t theEventBX;
  uint16_t theBX;
  uint16_t theData;
};

#include <iostream>
inline std::ostream& operator<<(std::ostream& o, const DTLocalTrigger& trig) {
  return o << " BX: " << trig.bx() << " quality: " << trig.quality();
}
#endif

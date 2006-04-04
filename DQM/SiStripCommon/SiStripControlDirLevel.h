#ifndef DQM_SiStripCommissioningSummary_SiStripControlDirLevel_H
#define DQM_SiStripCommissioningSummary_SiStripControlDirLevel_H

//@class SiStripControlDirLevel

//@author M.Wingham



#include <string>
#include <iostream>

using namespace std;

class SiStripControlDirLevel {

 public:

  /** Contructor. String expected in the form: "ControlView/FecSlotW/FecRingX/CCUaddressY/CCUchannelZ" or the equivalent path for any fec-slot/ccu-ring/ccu-address/ccu-channel.*/
  SiStripControlDirLevel(string);
  ~SiStripControlDirLevel();

  /** Method that interprets the control path string as a fec-slot, ccu-ring, ccu-address and ccu-channel. */
  void interpret();

  static unsigned short const all = 65535;

  unsigned short slot;
  unsigned short ring;
  unsigned short addr;
  unsigned short chan;

  private :

    string dir_;
};

#endif // DQM_SiStripCommissioningSummary_SiStripControlDirLevel_H

#ifndef DQM_SiStripCommon_SiStripControlDirPath_H
#define DQM_SiStripCommon_SiStripControlDirPath_H

#include <string>

using namespace std;

class SiStripControlDirPath {
  
 public:

  static unsigned short const all_ = 65535;

  SiStripControlDirPath() 
    : root_("/"),
    top_("SiStrip"),
    control_("ControlView"),
    crate_("FecCrate"),
    slot_("FecSlot"),
    ring_("FecRing"),
    addr_("CcuAddr"),
    chan_("CcuChan"),
    i2c_("I2cAddr"),
    sep_("/") {;}
  ~SiStripControlDirPath() {;}
  
  inline string top() { return root_ + top_; }
  inline string control() {return top_ + sep_ + control_; }
  string path( // unsigned short crate = all_,
	      unsigned short slot = all_,
	      unsigned short ring = all_,
	      unsigned short addr = all_,
	      unsigned short chan = all_
	      // unsigned short i2c = all_ 
	      );
  
 private:
  
  string root_;
  string top_;
  string control_;
  string crate_;
  string slot_;
  string ring_;
  string addr_;
  string chan_;
  string i2c_;
  string sep_;
  
};

#endif // DQM_SiStripCommon_SiStripControlDirPath_H

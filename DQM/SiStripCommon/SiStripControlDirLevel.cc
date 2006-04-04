#include "DQM/SiStripCommon/interface/SiStripControlDirLevel.h"
#include <stdlib.h>
#include <math.h>

SiStripControlDirLevel::SiStripControlDirLevel(string dir) :

  slot(all),
  ring(all),
  addr(all),
  chan(all),
  dir_(dir)
{;}
SiStripControlDirLevel::~SiStripControlDirLevel() {;}

void SiStripControlDirLevel::interpret() {

  if (dir_.compare(0, 12, "ControlView/")) {
  cout << "Warning directory not in \"ControlView/\"" << endl;}

  else {

    unsigned short index = 12;
    if (!dir_.compare(index, 7, "FecSlot")) {
      index += 7;
      unsigned short temp_index = dir_.find("/", index);
      string fecSlot(dir_,index,(temp_index - index));
      slot = atoi(fecSlot.c_str());
      index = dir_.find("/", index) + 1;
      
      if (!dir_.compare(index, 7, "FecRing")) {
	
	index += 7;
	unsigned short temp_index = dir_.find("/", index);
	string fecRing(dir_,index,(temp_index - index));
	ring = atoi(fecRing.c_str());
	index = dir_.find("/", index) + 1;
	
	if (!dir_.compare(index, 10, "CCUaddress")) {
	  
	index += 10;
	unsigned short temp_index = dir_.find("/", index);
	string ccuAddr(dir_,index,(temp_index - index));
	addr = atoi(ccuAddr.c_str());
	index = dir_.find("/", index) + 1;
	
	if (!dir_.compare(index, 10, "CCUchannel")) {
	  
	  index += 10;
	  unsigned short temp_index = dir_.find("/", index);
	  string ccuChan(dir_,index,(temp_index - index));
	  chan = atoi(ccuChan.c_str());	  
	}
      }
      }
    }
  }
}


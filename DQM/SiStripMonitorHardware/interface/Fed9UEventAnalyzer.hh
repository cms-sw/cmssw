#ifndef FED9UEVENTANALYZER_H
#define FED9UEVENTANALYZER_H

#include "DQM/SiStripMonitorHardware/interface/Fed9UDebugEvent.hh"

typedef struct Fed9UErrorCondition{
  // Generic (summary)
  int problemsSeen;
  int totalChannels;

  // FED-related
  bool internalFreeze;
  bool bxError;
  int apveAddress;

  // FPGA-related
  int feMajorAddress[8];
  bool feOverflow[8];
  bool feEnabled[8];
  bool apvAddressError[8];

  // Channel- (and APV-) related
  int apv[96*2];
  int channel[96];

} Fed9UErrorCondition;


class Fed9UEventAnalyzer {
private:
  // The FedEvent object
  Fed9U::Fed9UDebugEvent* fedEvent_; 
  
  // Lower and upper limits of the Tracker FED ids
  std::pair<int,int> fedIdBoundaries_;

  // Sets the swap of words of the FED buffer
  bool swapOn_;
  bool preSwapOn_; // bad hack TODO: clean this

  int thisFedId_;
  
public:
  // Constructor
  Fed9UEventAnalyzer(std::pair<int,int> newFedBoundaries, bool doSwap, bool doPreSwap);
  
  // Constructor and event analyzer
  Fed9UEventAnalyzer(Fed9U::u32* data_u32, Fed9U::u32 size_u32,
		     std::pair<int,int> newFedBoundaries, bool doSwap, bool doPreSwap);
  ~Fed9UEventAnalyzer();

  // The actual event analyzer
  bool Initialize(Fed9U::u32* data_u32, Fed9U::u32 size_u32);

  Fed9UErrorCondition Analyze();

  enum {APVERROR=0x01,
	APVWRONGHEADER=0x02,
	BADAPV=0x4,
	FIBEROUTOFSYNCH=0x8,
	FIBERUNLOCKED=0x10
  };

  int getFedId() {return thisFedId_; };
};

#endif

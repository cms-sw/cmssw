#ifndef TFHeaderDescription_h
#define TFHeaderDescription_h

#define FEDPMC_TYPE 0xDEF00001
#define FEDEMU_TYPE 0xDEF00003
#define FED9U_TYPE 0xDEF00002
#define FEDTRG_TYPE 0xDEF0DEF0
#include <cstdio>

class TFHeaderDescription {
private:
  unsigned long bunchCrossing_;
  unsigned long numberOfChannels_;
  unsigned long numberOfSamples_;
  unsigned long fedType_;
  unsigned long fedId_;
  unsigned long fedEventNumber_;

public:
  void setBunchCrossing(unsigned long t) { bunchCrossing_ = t; }
  void setNumberOfChannels(unsigned long t) { numberOfChannels_ = t; }
  void setNumberOfSamples(unsigned long t) { numberOfSamples_ = t; }
  void setFedType(unsigned long t) { fedType_ = t; }
  void setFedId(unsigned long t) { fedId_ = t; }
  void setFedEventNumber(unsigned long t) { fedEventNumber_ = t; }
  unsigned long getBunchCrossing() const { return bunchCrossing_; }
  unsigned long getNumberOfChannels() const { return numberOfChannels_; }
  unsigned long getNumberOfSamples() const { return numberOfSamples_; }
  unsigned long getFedType() const { return fedType_; }
  unsigned long getFedId() const { return fedId_; }
  unsigned long getFedEventNumber() const { return fedEventNumber_; }
  void Print() const {
    printf(
        "Bunch crossing %lx \n Number Of Channels %ld \n Number of Samples %ld \n Fed Type %lx \n Fed Id %lx \n Fed "
        "Event Number %ld \n",
        bunchCrossing_,
        numberOfChannels_,
        numberOfSamples_,
        fedType_,
        fedId_,
        fedEventNumber_);
  }
};

#endif

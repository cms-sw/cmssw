#ifndef CondFormats_SiStripObjects_Phase2TrackerModule_H
#define CondFormats_SiStripObjects_Phase2TrackerModule_H

#include <vector>
#include <algorithm>

#include "CondFormats/Serialization/interface/Serializable.h"

class Phase2TrackerModule {
public:
  enum ModuleTypes { SS, PS };

public:
  // normal constructor... for now, nothing is mandatory
  Phase2TrackerModule(ModuleTypes moduleType = SS,
                      uint32_t detid = 0,
                      uint32_t gbtid = 0,
                      uint32_t fedid = 0,
                      uint32_t fedch = 0,
                      uint32_t powerGroup = 0,
                      uint32_t coolingLoop = 0)
      : moduleType_(moduleType), detid_(detid), gbtid_(gbtid), powerGroup_(powerGroup), coolingLoop_(coolingLoop) {
    ch_ = std::make_pair(fedid, fedch);
  }

  // destructor
  virtual ~Phase2TrackerModule() {}

  // setters (complement the constructor)
  void setDetid(uint32_t detid) { detid_ = detid; }
  void setGbtid(uint32_t gbtid) { gbtid_ = gbtid; }
  void setFedChannel(unsigned int fedid, unsigned int fedch) { ch_ = std::make_pair(fedid, fedch); }
  void setCoolingLoop(uint32_t cl) { coolingLoop_ = cl; }
  void setPowerGroup(uint32_t pg) { powerGroup_ = pg; }
  void setModuleType(ModuleTypes moduleType) { moduleType_ = moduleType; }
  void addI2cDevice(unsigned int dev) { i2cDevices_.push_back(dev); }
  void setI2cDevices(std::vector<unsigned int> i2cd) { i2cDevices_ = i2cd; }

  // getters
  uint32_t getDetid() const { return detid_; }
  uint32_t getGbtid() const { return gbtid_; }
  std::pair<unsigned int, unsigned int> getCh() const { return ch_; }
  uint32_t getCoolingLoop() const { return coolingLoop_; }
  uint32_t getPowerGroup() const { return powerGroup_; }
  ModuleTypes getModuleType() const { return moduleType_; }
  const std::vector<unsigned int>& getI2cDevices() const { return i2cDevices_; }

  // description (for printing)
  std::string description(bool compact = false) const;

private:
  // the GBTid/fed map should be easy to build automatically, since the FED can ask to the link. It could be put in some commissioning packet
  // the detid/GBTid map comes from construction. It should be in the construction database
  // the power groups and cooling groups are defined in terms of detids. Known from construction.
  // ... of course, for now there is nothing like the GBTid...
  ModuleTypes moduleType_;
  uint32_t detid_, gbtid_;
  uint32_t powerGroup_, coolingLoop_;
  std::pair<unsigned int, unsigned int> ch_;
  std::vector<unsigned int> i2cDevices_;

  COND_SERIALIZABLE;
};

#endif

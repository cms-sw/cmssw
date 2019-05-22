#ifndef DataFormats_Phase2TrackerDigi_Phase2TrackerCommissioningDigi_H
#define DataFormats_Phase2TrackerDigi_Phase2TrackerCommissioningDigi_H

#include <cstdint>

class Phase2TrackerCommissioningDigi {
public:
  Phase2TrackerCommissioningDigi() : key_(0), value_(0) {}
  Phase2TrackerCommissioningDigi(uint32_t key, uint32_t value) : key_(key), value_(value) {}
  ~Phase2TrackerCommissioningDigi() {}
  uint32_t getKey() const { return key_; }
  uint32_t getValue() const { return value_; }

private:
  uint32_t key_;
  uint32_t value_;
};

#endif

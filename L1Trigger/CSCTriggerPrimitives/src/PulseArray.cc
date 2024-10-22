#include "L1Trigger/CSCTriggerPrimitives/interface/PulseArray.h"

PulseArray::PulseArray() {}

void PulseArray::initialize(unsigned numberOfChannels) {
  numberOfChannels_ = numberOfChannels;
  data_.clear();
  data_.resize(CSCConstants::NUM_LAYERS);
  for (unsigned layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
    data_[layer].resize(numberOfChannels);
  }
}

void PulseArray::clear() {
  // set all elements in the 2D vector to 0
  for (auto& p : data_) {
    for (auto& q : p) {
      q = 0;
    }
  }
}

unsigned& PulseArray::operator()(const unsigned layer, const unsigned channel) { return data_[layer][channel]; }

unsigned PulseArray::bitsInPulse() const { return 8 * sizeof(data_[0][0]); }

void PulseArray::extend(const unsigned layer, const unsigned channel, const unsigned bx, const unsigned hit_persist) {
  for (unsigned int ibx = bx; ibx < bx + hit_persist; ++ibx) {
    data_[layer][channel] = data_[layer][channel] | (1 << ibx);
  }
}

bool PulseArray::oneShotAtBX(const unsigned layer, const unsigned channel, const unsigned bx) const {
  return (data_[layer][channel] >> bx) & 1;
}

bool PulseArray::isOneShotHighAtBX(const unsigned layer, const unsigned channel, const unsigned bx) const {
  return oneShotAtBX(layer, channel, bx) == 1;
}

unsigned PulseArray::numberOfLayersAtBX(const unsigned bx) const {
  unsigned layers_hit = 0;
  for (unsigned layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
    for (unsigned channel = 0; channel < numberOfChannels_; channel++) {
      if (isOneShotHighAtBX(layer, channel, bx)) {
        layers_hit++;
        break;
      }
    }
  }
  return layers_hit;
}

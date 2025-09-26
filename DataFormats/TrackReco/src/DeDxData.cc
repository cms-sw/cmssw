#include "DataFormats/TrackReco/interface/DeDxData.h"

using namespace reco;

DeDxData::DeDxData() : value_(0.), error_(0.), numberOfMeasurements_(0), numberOfSatMeasurements_(0) { ; }

DeDxData::DeDxData(float val, int sat, unsigned int num)
    : value_(val), error_(-1.), numberOfMeasurements_(num), numberOfSatMeasurements_(sat) {
  ;
}

DeDxData::DeDxData(float val, float er, int sat, unsigned int num)
    : value_(val), error_(er), numberOfMeasurements_(num), numberOfSatMeasurements_(sat) {
  ;
}

DeDxData::~DeDxData() { ; }

float DeDxData::dEdx() const { return value_; }

float DeDxData::dEdxError() const { return error_; }

unsigned int DeDxData::numberOfMeasurements() const { return numberOfMeasurements_; }

int DeDxData::numberOfSaturatedMeasurements() const { return numberOfSatMeasurements_; }

//#include <math.h>
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"

using namespace reco;

TrackDeDxEstimate::TrackDeDxEstimate() : value_(0.), error_(0.), numberOfMeasurements_(0){}

TrackDeDxEstimate::TrackDeDxEstimate (float val, float er, unsigned int num) : value_(val), error_(er), numberOfMeasurements_(num) {}

TrackDeDxEstimate::~TrackDeDxEstimate(){}
float TrackDeDxEstimate::dEdx() const {return value_}
float TrackDeDxEstimate::dEdxError() const {return error_}

unsigned int TrackDeDxEstimate::numberOfMeasurements() const {return numberOfMeasurements_}


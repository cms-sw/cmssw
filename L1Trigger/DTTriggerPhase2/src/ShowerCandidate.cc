#include "L1Trigger/DTTriggerPhase2/interface/ShowerCandidate.h"

#include <cmath>
#include <iostream>
#include <memory>

using namespace cmsdt;

ShowerCandidate::ShowerCandidate() {
    clear();
}

ShowerCandidate& ShowerCandidate::operator=(const ShowerCandidate &other) {
    if (this != &other) {
        nhits_ = other.nhits_;
        rawId_ = other.rawId_;
        bx_ = other.bx_;
        wmin_ = other.wmin_;
        wmax_ = other.wmax_;
        avgPos_ = other.avgPos_;
        avgTime_ = other.avgTime_;
        shower_flag_ = other.shower_flag_;
    }
    return *this;
}

void ShowerCandidate::clear() {
    nhits_ = 0;
    bx_ = 0;
    wmin_ = 0;
    wmax_ = 0;
    avgPos_ = 0;
    avgTime_ = 0;
    shower_flag_ = false;
    wires_profile_.resize(96, 0);
}

#ifndef DATAFORMATS_PHASE2TRACKERCLUSTER_PHASE2TRACKERCLUSTER1D_H 
#define DATAFORMATS_PHASE2TRACKERCLUSTER_PHASE2TRACKERCLUSTER1D_H 

#include <stdint.h>

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

class Phase2TrackerCluster1D {

public:

    Phase2TrackerCluster1D() : data_(0) { }
    Phase2TrackerCluster1D(unsigned int row, unsigned int col, unsigned int size) : firstDigi_(row, col), data_((size & 0x7fff)) { }
    Phase2TrackerCluster1D(unsigned int row, unsigned int col, unsigned int size, unsigned int threshold) : firstDigi_(row, col), data_(((threshold & 0x1) << 15) | (size & 0x7fff)) { }
    Phase2TrackerCluster1D(const Phase2TrackerDigi& firstDigi, unsigned int size) : firstDigi_(firstDigi), data_((size & 0x7fff)) { }
    Phase2TrackerCluster1D(const Phase2TrackerDigi& firstDigi, unsigned int size, unsigned int threshold) : firstDigi_(firstDigi), data_(((threshold & 0x1) << 15) | (size & 0x7fff)) { }

    const Phase2TrackerDigi& firstDigi() const { return firstDigi_; }
    unsigned int firstStrip() const { return firstDigi_.strip(); }
    unsigned int firstRow() const { return firstDigi_.row(); }
    unsigned int edge() const { return firstDigi_.edge(); }
    unsigned int column() const { return firstDigi_.column(); }
    uint16_t size() const { return (data_ & 0x7fff); }
    uint16_t threshold() const { return ((data_ >> 15) & 0x1); }
    float center() const { return firstStrip() + (data_ & 0x7fff) / 2.; }
    std::pair< float, float > barycenter() const { return std::make_pair(column(), center()); }

private:

    Phase2TrackerDigi firstDigi_;
    uint16_t data_;

};  

inline bool operator< (const Phase2TrackerCluster1D& one, const Phase2TrackerCluster1D& other) {
    return one.firstStrip() < other.firstStrip();
}

typedef edmNew::DetSetVector< Phase2TrackerCluster1D > Phase2TrackerCluster1DCollectionNew;

#endif 

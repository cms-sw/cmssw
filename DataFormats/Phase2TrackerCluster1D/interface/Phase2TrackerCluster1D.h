#ifndef DATAFORMATS_PHASE2TRACKERCLUSTER1D_H 
#define DATAFORMATS_PHASE2TRACKERCLUSTER1D_H 

#include <stdint.h>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetRefVector.h"

#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

class Phase2TrackerCluster1D {

public:

    Phase2TrackerCluster1D() : size_(0) { }
    Phase2TrackerCluster1D(unsigned int row, unsigned int col, unsigned int size) : firstDigi_(row, col), size_(size) { }
    Phase2TrackerCluster1D(const Phase2TrackerDigi& firstDigi, unsigned int size) : firstDigi_(firstDigi), size_(size) { }

    const Phase2TrackerDigi& firstDigi() const { return firstDigi_; }
    unsigned int firstStrip() const { return firstDigi_.strip(); }
    unsigned int firstRow() const { return firstDigi_.row(); }
    unsigned int edge() const { return firstDigi_.edge(); }
    unsigned int column() const { return firstDigi_.column(); }
    uint16_t size() const { return size_; }
    float center() const { return firstStrip() + size_ / 2.; }
    std::pair< float, float > barycenter() const { return std::make_pair(column(), center()); }

private:

    Phase2TrackerDigi firstDigi_;
    uint16_t size_;

};  

inline bool operator< (const Phase2TrackerCluster1D& one, const Phase2TrackerCluster1D& other) {
    return one.center() < other.center();
}

typedef edm::DetSetVector< Phase2TrackerCluster1D > Phase2TrackerCluster1DCollection;
typedef edm::Ref< Phase2TrackerCluster1DCollection, Phase2TrackerCluster1D > Phase2TrackerCluster1DRef;
typedef edm::DetSetRefVector< Phase2TrackerCluster1D > Phase2TrackerCluster1DRefVector;
typedef edm::RefProd< Phase2TrackerCluster1DCollection > Phase2TrackerCluster1DRefProd;

typedef edmNew::DetSetVector< Phase2TrackerCluster1D > Phase2TrackerCluster1DCollectionNew;
typedef edm::Ref< Phase2TrackerCluster1DCollectionNew, Phase2TrackerCluster1D > Phase2TrackerCluster1DRefNew;

#endif 

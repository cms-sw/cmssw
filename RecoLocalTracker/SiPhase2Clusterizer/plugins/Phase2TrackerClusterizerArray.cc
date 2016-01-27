#include "RecoLocalTracker/SiPhase2Clusterizer/interface/Phase2TrackerClusterizerArray.h"

/*
 * Create a new Array of hits
 */

Phase2TrackerClusterizerArray::Phase2TrackerClusterizerArray() : nrows_(0), ncols_(0) { }

Phase2TrackerClusterizerArray::Phase2TrackerClusterizerArray(unsigned int nrows, unsigned int ncols) {
    setSize(nrows, ncols);
}

/*
 * Resize the Array and clear the values
 */

void Phase2TrackerClusterizerArray::setSize(unsigned int nrows, unsigned int ncols) {
    nrows_ = nrows;
    ncols_ = ncols;
    matrix_.resize(nrows * ncols);
    for (std::vector< bool >::iterator it(matrix_.begin()); it != matrix_.end(); ++it) *it = false;
}

/*
 * Return the value of an element in the Array
 */

bool Phase2TrackerClusterizerArray::operator()(unsigned int row, unsigned int col) const {
    if (inside(row, col)) return matrix_[index(row, col)];
    else return false;
}

/*
 * Return the number of rows
 */

unsigned int Phase2TrackerClusterizerArray::rows() const { 
    return nrows_; 
}

/*
 * Return the number of columns
 */

unsigned int Phase2TrackerClusterizerArray::columns() const { 
    return ncols_; 
}

/*
 * Tell if an element is inside the array or not
 */

bool Phase2TrackerClusterizerArray::inside(unsigned int row, unsigned int col) const {
    return (row < nrows_ && col < ncols_);
}

/*
 * Change the value of an element of the Array
 */

void Phase2TrackerClusterizerArray::set(unsigned int row, unsigned int col, bool state) {
    matrix_[index(row, col)] = state;
}

/*
 * Return the size of the array
 */

unsigned int Phase2TrackerClusterizerArray::size() const { 
    return matrix_.size(); 
}

/*
 * Get the poisiton of an element of the Array in the vector that holds the data
 */

unsigned int Phase2TrackerClusterizerArray::index(unsigned int row, unsigned int col) const { 
    return col * nrows_ + row; 
}


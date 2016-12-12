#ifndef RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizerArray_h
#define RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizerArray_h

#include <vector>

class Phase2TrackerClusterizerArray {

    public:

        inline Phase2TrackerClusterizerArray();
        inline Phase2TrackerClusterizerArray(unsigned int, unsigned int);
//         inline void setSize(unsigned int, unsigned int);
        inline int operator()(unsigned int, unsigned int) const;
        inline unsigned int rows() const;
        inline unsigned int columns() const;
        inline bool inside(unsigned int, unsigned int) const;
        inline void set(unsigned int, unsigned int, bool, bool);
        inline unsigned int size() const;
        inline unsigned int index(unsigned int, unsigned int) const;

    private:

        unsigned int nrows_, ncols_;
        std::vector< bool > matrix_;
        std::vector< bool > hipmatrix_;

};


/*
 * Create a new Array of hits
 */

Phase2TrackerClusterizerArray::Phase2TrackerClusterizerArray() : nrows_(0), ncols_(0) { }

Phase2TrackerClusterizerArray::Phase2TrackerClusterizerArray(unsigned int nrows, unsigned int ncols) :
    nrows_(nrows),
    ncols_(ncols),
    matrix_(nrows * ncols,false),
    hipmatrix_(nrows * ncols,false){}


/*
 * Return the value of an element in the Array
 */

int Phase2TrackerClusterizerArray::operator()(unsigned int row, unsigned int col) const {
   if (!inside(row, col)) return 0; // FIXME this should go outside: avoid it
   return matrix_[index(row, col)]  ? 
              ( hipmatrix_[index(row, col)] ? 2 : 1 ) : 0;
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
    return ( (row < nrows_) & (col < ncols_) );
}

/*
 * Change the value of an element of the Array
 */

void Phase2TrackerClusterizerArray::set(unsigned int row, unsigned int col, bool state, bool hip) {
    matrix_[index(row, col)] = state;
    hipmatrix_[index(row, col)] = hip;
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

#endif

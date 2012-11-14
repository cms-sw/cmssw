#ifndef NPSTAT_ARRAYNDSCANNER_HH_
#define NPSTAT_ARRAYNDSCANNER_HH_

/*!
// \file ArrayNDScanner.h
//
// \brief Iteration over indices of a multidimensional array
//
// Author: I. Volobouev
//
// July 2012
*/

#include <vector>
#include <climits>

namespace npstat {
   /**
    * This class can be used to iterate over array indices without actually
    * building the array or requesting any memory from the heap. Typical use:
    *
    * @code
    *   for (ArrayNDScanner scanner(shape); scanner.isValid(); ++scanner)
    *   {
    *       scanner.getIndex(indexArray, indexArrayLen);
    *       .... Do what is necessary with multidimensional index ....
    *       .... Extract linear index: ...............................
    *       scanner.state();
    *   }
    * @endcode
    *
    * This can be useful, for example, in case one needs to iterate over
    * slices of some array (so that the array itself can not be used
    * to obtain similar information easily).
    */
    class ArrayNDScanner
    {
    public:
        //@{
        /** Constructor from a multidimensional array shape */
        inline ArrayNDScanner(const unsigned* shape, const unsigned lenShape)
            {initialize(shape, lenShape);}

        inline ArrayNDScanner(const std::vector<unsigned>& shape)
            {initialize(shape.empty() ? static_cast<unsigned*>(0) : 
                        &shape[0], shape.size());}
        //@}

        /** Dimensionality of the scan */
        inline unsigned dim() const {return dim_;}

        /** Retrieve current state (i.e., linear index of the scan) */
        inline unsigned long state() const {return state_;}

        /** Maximum possible state (i.e., linear index of the scan) */
        inline unsigned long maxState() const {return maxState_;}

        /** Returns false when iteration is complete */
        inline bool isValid() const {return state_ < maxState_;}

        /** Retrieve current multidimensional index */
        void getIndex(unsigned* index, unsigned indexBufferLen) const;

        /** Reset the state (as if the object has just been constructed) */
        inline void reset() {state_ = 0UL;}

        /** Prefix increment */
        inline ArrayNDScanner& operator++()
            {if (state_ < maxState_) ++state_; return *this;}

        /** Postfix increment (distinguished by the dummy "int" parameter) */
        inline void operator++(int) {if (state_ < maxState_) ++state_;}

        /** Set the state directly */
        inline void setState(const unsigned long state)
            {state_ = state <= maxState_ ? state : maxState_;}

    private:
        ArrayNDScanner();
        
        void initialize(const unsigned* shape, unsigned lenShape);

        unsigned long strides_[CHAR_BIT*sizeof(unsigned long)];
        unsigned long state_;
        unsigned long maxState_;
        unsigned dim_;
    };
}

#endif // NPSTAT_ARRAYSCANNER_HH_


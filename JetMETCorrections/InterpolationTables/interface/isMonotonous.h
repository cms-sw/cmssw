#ifndef NPSTAT_ISMONOTONOUS_HH_
#define NPSTAT_ISMONOTONOUS_HH_

/*!
// \file isMonotonous.h
//
// \brief A few simple template functions for checking monotonicity of
//        container values
//
// Author: I. Volobouev
//
// July 2012
*/

namespace npstat {
    /** Check if the sequence of values is strictly increasing */
    template<class Iter>
    inline bool isStrictlyIncreasing(Iter begin, Iter const end)
    {
        if (begin == end)
            return false;
        Iter first(begin);
        bool status = ++begin != end;
        for (; begin != end && status; ++begin, ++first)
            if (!(*first < *begin))
                status = false;
        return status;
    }

    /** Check if the sequence of values is strictly decreasing */
    template<class Iter>
    inline bool isStrictlyDecreasing(Iter begin, Iter const end)
    {
        if (begin == end)
            return false;
        Iter first(begin);
        bool status = ++begin != end;
        for (; begin != end && status; ++begin, ++first)
            if (!(*begin < *first))
                status = false;
        return status;
    }

    /** Check if the sequence of values is strictly increasing or decreasing */
    template<class Iter>
    inline bool isStrictlyMonotonous(Iter const begin, Iter const end)
    {
        return isStrictlyIncreasing(begin, end) ||
               isStrictlyDecreasing(begin, end);
    }

    /** Check if the sequence of values is not decreasing */
    template<class Iter>
    inline bool isNonDecreasing(Iter begin, Iter const end)
    {
        if (begin == end)
            return false;
        Iter first(begin);
        bool status = ++begin != end;
        for (; begin != end && status; ++begin, ++first)
            if (*begin < *first)
                status = false;
        return status;
    }

    /** Check if the sequence of values is not increasing */
    template<class Iter>
    inline bool isNonIncreasing(Iter begin, Iter const end)
    {
        if (begin == end)
            return false;
        Iter first(begin);
        bool status = ++begin != end;
        for (; begin != end && status; ++begin, ++first)
            if (*begin > *first)
                status = false;
        return status;
    }

    /** 
    // Check if the sequence of values is either non-increasing
    // or non-decreasing
    */
    template<class Iter>
    inline bool isMonotonous(Iter const begin, Iter const end)
    {
        return isNonDecreasing(begin, end) || isNonIncreasing(begin, end);
    }
}

#endif // NPSTAT_ISMONOTONOUS_HH_


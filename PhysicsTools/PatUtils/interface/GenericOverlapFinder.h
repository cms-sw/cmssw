#ifndef PhysicsTools_PatUtils_GenericOverlapFinder_h
#define PhysicsTools_PatUtils_GenericOverlapFinder_h

#include "DataFormats/Math/interface/deltaR.h"

#include <memory>
#include <vector>
#include <algorithm>

namespace pat {

    /// A vector of pairs of indices <i1,i2>, for each i1 that overlaps, 
    /// i2 is the "best" overlap match.
    /// if an element of does not overlap with anyone, it will not end up in this list.
    typedef std::vector< std::pair<size_t, size_t> > OverlapList;

    /// Turn a comparator in a distance for uses of overlap
    ///    comparator(x,y) = true <-> x and y overlap
    /// the associated distance would work as
    ///     dist(x,y) = 1.0 - comparator(x,y)   [0 if overlap, 1 otherwise]
    /// if the constructor of your Comparator does not need parameters, 
    /// then the associated distance will not need parameters too.
    template<typename Comparator>
    struct OverlapDistance {
        public:
            OverlapDistance() {}
            OverlapDistance(const Comparator &comp) : comp_(comp) {}
            template<typename T1, typename T2>
            double operator()(const T1 &t1, const T2 &t2) const {
                return 1.0 - comp_(t1,t2);
            }
        private:
            Comparator comp_;
    }; //struct

    /// Distance with deltaR metrics and a fixed maximum for the overlap deltaR
    ///   dist(x,y) = deltaR2(x,y) / deltaR2cut;
    struct OverlapByDeltaR {
        public:
            OverlapByDeltaR(double deltaR) :  scale_(1.0/(deltaR*deltaR)) {}
            template<typename T1, typename T2>
            double operator()(const T1 &t1, const T2 &t2) const {
                return deltaR2(t1,t2) * scale_;
            }
        private:
            double scale_;
    }; //struct


    template <typename Distance>
    class GenericOverlapFinder {

        public:

           GenericOverlapFinder() {}
           GenericOverlapFinder(const Distance &dist) : distance_(dist) {}
            
            /// Indices of overlapped items, and of the nearest item on they overlap with
            /// Items are considered to overlap if distance(x1,x2) < 1
            /// both Collections can be vectors, Views, or anything with the same interface
            template <typename Collection, typename OtherCollection>
            std::auto_ptr< OverlapList >
            find(const Collection &items, const OtherCollection &other) const ;

        private:
            Distance distance_;

    }; // class
}

template<typename Distance>
template<typename Collection, typename OtherCollection>
std::auto_ptr< pat::OverlapList >
pat::GenericOverlapFinder<Distance>::find(const Collection &items, const OtherCollection &other) const 
{
    size_t size = items.size(), size2 = other.size();

    std::auto_ptr< OverlapList > ret(new OverlapList());
    
    for (size_t ie = 0; ie < size; ++ie) {
        double dmin   = 1.0;
        size_t match = 0;

        for (size_t je = 0; je < size2; ++je) {
            double dist = distance_(items[ie], other[je]);
            if (dist < dmin) { match = je; dmin = dist;  }
        }
        
        if (dmin < 1.0) {
            ret->push_back(std::make_pair(ie,match));
        }
    }

    return ret;
}


#endif

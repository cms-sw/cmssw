#ifndef PhysicsTools_PatUtils_GenericDuplicateRemover_h
#define PhysicsTools_PatUtils_GenericDuplicateRemover_h

#include <memory>
#include <vector>
#include <sys/types.h>

namespace pat {

    template <typename Comparator, typename Arbitrator>
    class GenericDuplicateRemover {

        public:

           GenericDuplicateRemover() {}
           GenericDuplicateRemover(const Comparator &comp) : comparator_(comp) {}
           GenericDuplicateRemover(const Comparator &comp, const Arbitrator &arbiter) : comparator_(comp), arbiter_(arbiter) {}
            
           ~GenericDuplicateRemover() {}

            /// Indices of duplicated items to remove
            /// Comparator is used to check for duplication, Arbiter to pick the best one
            /// e.g. comparator(x1, x2) should return true if they are duplicates
            ///      arbitrator(x1, x2) should return true if x1 is better, that is we want to keep x1 and delete x2
            /// Collection can be vector, View, or anything with the same interface
            template <typename Collection>
            std::auto_ptr< std::vector<size_t> >
            duplicates(const Collection &items) const ;

        private:
            Comparator comparator_;
            Arbitrator   arbiter_;

    }; // class
}

template<typename Comparator, typename Arbitrator>
template<typename Collection>
std::auto_ptr< std::vector<size_t> >
pat::GenericDuplicateRemover<Comparator,Arbitrator>::duplicates(const Collection &items) const 
{
    size_t size = items.size();

    std::vector<bool> bad(size, false);

    for (size_t ie = 0; ie < size; ++ie) {
        if (bad[ie]) continue; // if already marked bad

        for (size_t je = ie+1; je < size; ++je) {

            if (bad[je]) continue; // if already marked bad

            if ( comparator_(items[ie], items[je]) ) {
                int toRemove = arbiter_(items[ie], items[je]) ? je : ie;
                bad[toRemove] = true;
            }
        }
    }

    std::auto_ptr< std::vector<size_t> > ret(new std::vector<size_t>());

    for (size_t i = 0; i < size; ++i) {
        if (bad[i]) ret->push_back(i);
    }

    return ret;
}


#endif

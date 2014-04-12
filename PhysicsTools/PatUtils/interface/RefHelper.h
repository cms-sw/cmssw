#ifndef PhysicsTools_PatUtils_RefHelper_h
#define PhysicsTools_PatUtils_RefHelper_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/ValueMap.h"
namespace pat { namespace helper {
/*** \brief A class to help manage references and associative containers in some tricky cases (e.g.  selection by copy)
*/
template<typename T>
class RefHelper {
    public:
        typedef typename edm::Ptr<T> Ref;

        /// Constructor taking a ValueMap of back-references daughter => mother.
        RefHelper(const edm::ValueMap< Ref > &backRefMap) : backRefMap_(backRefMap) { }

        /// Returns a Ref to the direct parent of "ref", or a null Ref if "ref" is already root
        Ref parentOrNull(const Ref &ref) const ;

        /// Returns a Ref to the direct parent of "ref", or "ref" itself if it's already root
        Ref parentOrSelf(const Ref &ref) const ;

        /// Climbs back the Ref chain and returns the root of the branch starting from "ref"
        Ref ancestorOrSelf(const Ref &ref) const ;

        bool isRoot(const Ref &ref) const ;

        /// true if old is some ancestor of young (it does not have to be the root)
        bool isAncestorOf(const Ref &old, const Ref &young) const ;

        /// true if the two share the same root
        bool sharedAncestor(const Ref &ref1, const Ref &ref2) const ;

        /// Recursively looks up map to find something associated to ref, or one of its parents. 
        /// Throws edm::Exception(edm::errors::InvalidReference) if there's no match
        template<typename V, typename SomeRef>
        V recursiveLookup(const SomeRef &ref, const edm::ValueMap<V> &map) const ;

        /// Looks up map to find something associated to the root ancestor of "ref"
        /// Throws edm::Exception(edm::errors::InvalidReference) if there's no match
        template<typename V, typename SomeRef>
        V ancestorLookup(const SomeRef &ref, const edm::ValueMap<V> &map) const ;

    private:
        const edm::ValueMap< Ref > & backRefMap_;
};

template<typename T>
typename RefHelper<T>::Ref RefHelper<T>::parentOrNull(const RefHelper<T>::Ref &ref) const { 
    if (backRefMap_.contains(ref.id())) {
        try {
            return backRefMap_[ref];
        } catch (edm::Exception &e) {
            if (e.categoryCode() == edm::errors::InvalidReference) {
                return Ref();
            } else {
                throw;
            }
        }
    } else {
        return Ref();
    }
}

template<typename T>
typename RefHelper<T>::Ref RefHelper<T>::parentOrSelf(const RefHelper<T>::Ref &ref) const {
    Ref ret = parentOrNull(ref);
    return ret.isNonnull() ? ret : ref;
}

template<typename T>
typename RefHelper<T>::Ref RefHelper<T>::ancestorOrSelf(const RefHelper<T>::Ref &ref) const {
    Ref ret = ref;
    do {
        Ref test = parentOrNull(ret);
        if (test.isNull()) return ret;
        ret = test;
    } while (true);
}


template<typename T>
bool RefHelper<T>::isRoot(const RefHelper<T>::Ref &ref) const { 
    return parentOrNull(ref).isNull(); 
}

template<typename T>
bool RefHelper<T>::isAncestorOf(const RefHelper<T>::Ref &old, const RefHelper<T>::Ref &young) const { 
    Ref test = young;
    do {
        if (test == old) return true;
        test = parentOrNull(test);
    } while (test.isNonnull());
    return false;
}

template<typename T>
bool RefHelper<T>::sharedAncestor(const RefHelper<T>::Ref &ref1, const RefHelper<T>::Ref &ref2) const { 
    return ( ancestorOrSelf(ref1) == ancestorOrSelf(ref2) );
}

template<typename T>
template<typename V, typename SomeRef>
V RefHelper<T>::recursiveLookup(const SomeRef &ref, const edm::ValueMap<V> &map) const {
    Ref test(ref);
    do {
        if (map.contains(test.id())) {
            try {
                return map[test];
            } catch (edm::Exception &e) {
                if (e.categoryCode() != edm::errors::InvalidReference) {
                    throw;
                }
            }
        } 
        test = parentOrNull(test);
    } while (test.isNonnull());
    throw edm::Exception(edm::errors::InvalidReference) <<
        "RefHelper: recursive Lookup failed: neither the specified ref nor any of its parents are in the map.\n";
}

template<typename T>
template<typename V, typename SomeRef>
V RefHelper<T>::ancestorLookup(const SomeRef &ref, const edm::ValueMap<V> &map) const {
    Ref tref(ref);
    return map[ancestorOrSelf(tref)];
}

} } // namespace
#endif

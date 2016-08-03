#ifndef CondFormats_HcalObjects_AbsHFPhase1AlgoData_h_
#define CondFormats_HcalObjects_AbsHFPhase1AlgoData_h_

#include <typeinfo>

#include "boost/serialization/base_object.hpp"
#include "boost/serialization/export.hpp"

// Archive headers are needed here for the serialization registration to work.
// <cassert> is needed for the archive headers to work.
#if !defined(__GCCXML__)
#include <cassert>
#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"
#endif /* #if !defined(__GCCXML__) */

//
// Classes inheriting from this one are supposed to configure algorithms
// that inherit from AbsHFPhase1Algo (in package RecoLocalCalo/HcalRecAlgos)
//
class AbsHFPhase1AlgoData
{
public:
    inline virtual ~AbsHFPhase1AlgoData() {}

    // Comparison operators. Note that they are not virtual and should
    // not be overriden by derived classes. These operators are very
    // useful for I/O testing.
    inline bool operator==(const AbsHFPhase1AlgoData& r) const
        {return (typeid(*this) == typeid(r)) && this->isEqual(r);}
    inline bool operator!=(const AbsHFPhase1AlgoData& r) const
        {return !(*this == r);}

protected:
    // Method needed to compare objects for equality.
    // Must be implemented by derived classes.
    virtual bool isEqual(const AbsHFPhase1AlgoData&) const = 0;

private:
    friend class boost::serialization::access;
    template <typename Ar> 
    inline void serialize(Ar& ar, unsigned /* version */) {}
};

BOOST_CLASS_EXPORT_KEY(AbsHFPhase1AlgoData)

#endif // CondFormats_HcalObjects_AbsHFPhase1AlgoData_h_

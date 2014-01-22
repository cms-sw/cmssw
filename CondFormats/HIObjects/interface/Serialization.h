#ifndef CondFormats_HIObjects_Serialization_H
#define CondFormats_HIObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void CentralityTable::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_table);
}

template <class Archive>
void CentralityTable::BinValues::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mean);
    ar & BOOST_SERIALIZATION_NVP(var);
}

template <class Archive>
void CentralityTable::CBin::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(bin_edge);
    ar & BOOST_SERIALIZATION_NVP(n_part);
    ar & BOOST_SERIALIZATION_NVP(n_coll);
    ar & BOOST_SERIALIZATION_NVP(n_hard);
    ar & BOOST_SERIALIZATION_NVP(b);
    ar & BOOST_SERIALIZATION_NVP(eccRP);
    ar & BOOST_SERIALIZATION_NVP(ecc2);
    ar & BOOST_SERIALIZATION_NVP(ecc3);
    ar & BOOST_SERIALIZATION_NVP(ecc4);
    ar & BOOST_SERIALIZATION_NVP(ecc5);
    ar & BOOST_SERIALIZATION_NVP(S);
    ar & BOOST_SERIALIZATION_NVP(var0);
    ar & BOOST_SERIALIZATION_NVP(var1);
    ar & BOOST_SERIALIZATION_NVP(var2);
}

template <class Archive>
void RPFlatParams::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_table);
}

template <class Archive>
void RPFlatParams::EP::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(x);
    ar & BOOST_SERIALIZATION_NVP(y);
    ar & BOOST_SERIALIZATION_NVP(xSub1);
    ar & BOOST_SERIALIZATION_NVP(ySub1);
    ar & BOOST_SERIALIZATION_NVP(xSub2);
    ar & BOOST_SERIALIZATION_NVP(ySub2);
    ar & BOOST_SERIALIZATION_NVP(RPNameIndx);
}

namespace cond {
namespace serialization {

template <>
struct access<CentralityTable>
{
    static bool equal_(const CentralityTable & first, const CentralityTable & second)
    {
        return true
            and (equal(first.m_table, second.m_table))
        ;
    }
};

template <>
struct access<CentralityTable::BinValues>
{
    static bool equal_(const CentralityTable::BinValues & first, const CentralityTable::BinValues & second)
    {
        return true
            and (equal(first.mean, second.mean))
            and (equal(first.var, second.var))
        ;
    }
};

template <>
struct access<CentralityTable::CBin>
{
    static bool equal_(const CentralityTable::CBin & first, const CentralityTable::CBin & second)
    {
        return true
            and (equal(first.bin_edge, second.bin_edge))
            and (equal(first.n_part, second.n_part))
            and (equal(first.n_coll, second.n_coll))
            and (equal(first.n_hard, second.n_hard))
            and (equal(first.b, second.b))
            and (equal(first.eccRP, second.eccRP))
            and (equal(first.ecc2, second.ecc2))
            and (equal(first.ecc3, second.ecc3))
            and (equal(first.ecc4, second.ecc4))
            and (equal(first.ecc5, second.ecc5))
            and (equal(first.S, second.S))
            and (equal(first.var0, second.var0))
            and (equal(first.var1, second.var1))
            and (equal(first.var2, second.var2))
        ;
    }
};

template <>
struct access<RPFlatParams>
{
    static bool equal_(const RPFlatParams & first, const RPFlatParams & second)
    {
        return true
            and (equal(first.m_table, second.m_table))
        ;
    }
};

template <>
struct access<RPFlatParams::EP>
{
    static bool equal_(const RPFlatParams::EP & first, const RPFlatParams::EP & second)
    {
        return true
            and (equal(first.x, second.x))
            and (equal(first.y, second.y))
            and (equal(first.xSub1, second.xSub1))
            and (equal(first.ySub1, second.ySub1))
            and (equal(first.xSub2, second.xSub2))
            and (equal(first.ySub2, second.ySub2))
            and (equal(first.RPNameIndx, second.RPNameIndx))
        ;
    }
};

}
}

#endif

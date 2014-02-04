
#include "CondFormats/HIObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void CentralityTable::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_table);
}
COND_SERIALIZATION_INSTANTIATE(CentralityTable);

template <class Archive>
void CentralityTable::BinValues::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mean);
    ar & BOOST_SERIALIZATION_NVP(var);
}
COND_SERIALIZATION_INSTANTIATE(CentralityTable::BinValues);

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
COND_SERIALIZATION_INSTANTIATE(CentralityTable::CBin);

template <class Archive>
void RPFlatParams::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_table);
}
COND_SERIALIZATION_INSTANTIATE(RPFlatParams);

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
COND_SERIALIZATION_INSTANTIATE(RPFlatParams::EP);

#include "CondFormats/HIObjects/src/SerializationManual.h"

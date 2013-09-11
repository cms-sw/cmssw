#ifndef CondFormats_Calibration_Serialization_H
#define CondFormats_Calibration_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void Algo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(a);
}

template <class Archive>
void AlgoMap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("std::map<std::string, Algo>", boost::serialization::base_object<std::map<std::string, Algo>>(*this));
}

template <class Archive>
void Algob::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(b);
}

template <class Archive>
void BlobComplex::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(objects);
}

template <class Archive>
void BlobComplexContent::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data1);
    ar & BOOST_SERIALIZATION_NVP(data2);
    ar & BOOST_SERIALIZATION_NVP(data3);
}

template <class Archive>
void BlobComplexData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(a);
    ar & BOOST_SERIALIZATION_NVP(b);
    ar & BOOST_SERIALIZATION_NVP(values);
}

template <class Archive>
void BlobComplexObjects::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(a);
    ar & BOOST_SERIALIZATION_NVP(b);
    ar & BOOST_SERIALIZATION_NVP(content);
}

template <class Archive>
void BlobNoises::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_noises);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}

template <class Archive>
void BlobNoises::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}

template <class Archive>
void BlobPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_pedestals);
}

template <class Archive>
void CalibHistogram::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_histo);
}

template <class Archive>
void CalibHistograms::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_data);
}

template <class Archive>
void Pedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_pedestals);
}

template <class Archive>
void Pedestals::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_mean);
    ar & BOOST_SERIALIZATION_NVP(m_variance);
}

template <class Archive>
void big::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(tVector_);
    ar & BOOST_SERIALIZATION_NVP(thVector_);
    ar & BOOST_SERIALIZATION_NVP(sVector_);
    ar & BOOST_SERIALIZATION_NVP(id_current);
    ar & BOOST_SERIALIZATION_NVP(index_id);
    ar & BOOST_SERIALIZATION_NVP(cota_current);
    ar & BOOST_SERIALIZATION_NVP(cotb_current);
    ar & BOOST_SERIALIZATION_NVP(abs_cotb);
    ar & BOOST_SERIALIZATION_NVP(fpix_current);
}

template <class Archive>
void big::bigEntry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(runnum);
    ar & BOOST_SERIALIZATION_NVP(alpha);
    ar & BOOST_SERIALIZATION_NVP(cotalpha);
    ar & BOOST_SERIALIZATION_NVP(beta);
    ar & BOOST_SERIALIZATION_NVP(cotbeta);
    ar & BOOST_SERIALIZATION_NVP(costrk);
    ar & BOOST_SERIALIZATION_NVP(qavg);
    ar & BOOST_SERIALIZATION_NVP(symax);
    ar & BOOST_SERIALIZATION_NVP(dyone);
    ar & BOOST_SERIALIZATION_NVP(syone);
    ar & BOOST_SERIALIZATION_NVP(sxmax);
    ar & BOOST_SERIALIZATION_NVP(dxone);
    ar & BOOST_SERIALIZATION_NVP(sxone);
    ar & BOOST_SERIALIZATION_NVP(dytwo);
    ar & BOOST_SERIALIZATION_NVP(sytwo);
    ar & BOOST_SERIALIZATION_NVP(dxtwo);
    ar & BOOST_SERIALIZATION_NVP(sxtwo);
    ar & BOOST_SERIALIZATION_NVP(qmin);
    ar & BOOST_SERIALIZATION_NVP(par);
    ar & BOOST_SERIALIZATION_NVP(ytemp);
    ar & BOOST_SERIALIZATION_NVP(xtemp);
    ar & BOOST_SERIALIZATION_NVP(avg);
    ar & BOOST_SERIALIZATION_NVP(aqfl);
    ar & BOOST_SERIALIZATION_NVP(chi2);
    ar & BOOST_SERIALIZATION_NVP(spare);
}

template <class Archive>
void big::bigHeader::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(title);
    ar & BOOST_SERIALIZATION_NVP(ID);
    ar & BOOST_SERIALIZATION_NVP(NBy);
    ar & BOOST_SERIALIZATION_NVP(NByx);
    ar & BOOST_SERIALIZATION_NVP(NBxx);
    ar & BOOST_SERIALIZATION_NVP(NFy);
    ar & BOOST_SERIALIZATION_NVP(NFyx);
    ar & BOOST_SERIALIZATION_NVP(NFxx);
    ar & BOOST_SERIALIZATION_NVP(vbias);
    ar & BOOST_SERIALIZATION_NVP(temperature);
    ar & BOOST_SERIALIZATION_NVP(fluence);
    ar & BOOST_SERIALIZATION_NVP(qscale);
    ar & BOOST_SERIALIZATION_NVP(s50);
    ar & BOOST_SERIALIZATION_NVP(templ_version);
}

template <class Archive>
void big::bigStore::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(head);
    ar & BOOST_SERIALIZATION_NVP(entby);
    ar & BOOST_SERIALIZATION_NVP(entbx);
    ar & BOOST_SERIALIZATION_NVP(entfy);
    ar & BOOST_SERIALIZATION_NVP(entfx);
}

template <class Archive>
void boostTypeObj::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(a);
    ar & BOOST_SERIALIZATION_NVP(b);
    ar & BOOST_SERIALIZATION_NVP(aa);
    ar & BOOST_SERIALIZATION_NVP(bb);
}

template <class Archive>
void child::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("mybase", boost::serialization::base_object<mybase>(*this));
    ar & BOOST_SERIALIZATION_NVP(b);
}

template <class Archive>
void condex::ConfF::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::BaseKeyed", boost::serialization::base_object<cond::BaseKeyed>(*this));
    ar & BOOST_SERIALIZATION_NVP(v);
    ar & BOOST_SERIALIZATION_NVP(key);
}

template <class Archive>
void condex::ConfI::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::BaseKeyed", boost::serialization::base_object<cond::BaseKeyed>(*this));
    ar & BOOST_SERIALIZATION_NVP(v);
    ar & BOOST_SERIALIZATION_NVP(key);
}

template <class Archive>
void condex::Efficiency::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void condex::ParametricEfficiencyInEta::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("condex::Efficiency", boost::serialization::base_object<condex::Efficiency>(*this));
    ar & BOOST_SERIALIZATION_NVP(cutLow);
    ar & BOOST_SERIALIZATION_NVP(cutHigh);
    ar & BOOST_SERIALIZATION_NVP(low);
    ar & BOOST_SERIALIZATION_NVP(high);
}

template <class Archive>
void condex::ParametricEfficiencyInPt::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("condex::Efficiency", boost::serialization::base_object<condex::Efficiency>(*this));
    ar & BOOST_SERIALIZATION_NVP(cutLow);
    ar & BOOST_SERIALIZATION_NVP(cutHigh);
    ar & BOOST_SERIALIZATION_NVP(low);
    ar & BOOST_SERIALIZATION_NVP(high);
}

template <class Archive>
void fakeMenu::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_algorithmMap);
}

template <typename T, unsigned int S>
template <class Archive>
void fixedArray<T, S>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(content);
}

template <class Archive>
void mySiStripNoises::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_noises);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}

template <class Archive>
void mySiStripNoises::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}

template <class Archive>
void mybase::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void mypt::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pt);
}

template <class Archive>
void strKeyMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_content);
}

namespace cond {
namespace serialization {

template <>
struct access<Algo>
{
    static bool equal_(const Algo & first, const Algo & second)
    {
        return true
            and (equal(first.a, second.a))
        ;
    }
};

template <>
struct access<AlgoMap>
{
    static bool equal_(const AlgoMap & first, const AlgoMap & second)
    {
        return true
            and (equal(static_cast<const std::map<std::string, Algo> &>(first), static_cast<const std::map<std::string, Algo> &>(second)))
        ;
    }
};

template <>
struct access<Algob>
{
    static bool equal_(const Algob & first, const Algob & second)
    {
        return true
            and (equal(first.b, second.b))
        ;
    }
};

template <>
struct access<BlobComplex>
{
    static bool equal_(const BlobComplex & first, const BlobComplex & second)
    {
        return true
            and (equal(first.objects, second.objects))
        ;
    }
};

template <>
struct access<BlobComplexContent>
{
    static bool equal_(const BlobComplexContent & first, const BlobComplexContent & second)
    {
        return true
            and (equal(first.data1, second.data1))
            and (equal(first.data2, second.data2))
            and (equal(first.data3, second.data3))
        ;
    }
};

template <>
struct access<BlobComplexData>
{
    static bool equal_(const BlobComplexData & first, const BlobComplexData & second)
    {
        return true
            and (equal(first.a, second.a))
            and (equal(first.b, second.b))
            and (equal(first.values, second.values))
        ;
    }
};

template <>
struct access<BlobComplexObjects>
{
    static bool equal_(const BlobComplexObjects & first, const BlobComplexObjects & second)
    {
        return true
            and (equal(first.a, second.a))
            and (equal(first.b, second.b))
            and (equal(first.content, second.content))
        ;
    }
};

template <>
struct access<BlobNoises>
{
    static bool equal_(const BlobNoises & first, const BlobNoises & second)
    {
        return true
            and (equal(first.v_noises, second.v_noises))
            and (equal(first.indexes, second.indexes))
        ;
    }
};

template <>
struct access<BlobNoises::DetRegistry>
{
    static bool equal_(const BlobNoises::DetRegistry & first, const BlobNoises::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
            and (equal(first.iend, second.iend))
        ;
    }
};

template <>
struct access<BlobPedestals>
{
    static bool equal_(const BlobPedestals & first, const BlobPedestals & second)
    {
        return true
            and (equal(first.m_pedestals, second.m_pedestals))
        ;
    }
};

template <>
struct access<CalibHistogram>
{
    static bool equal_(const CalibHistogram & first, const CalibHistogram & second)
    {
        return true
            and (equal(first.m_histo, second.m_histo))
        ;
    }
};

template <>
struct access<CalibHistograms>
{
    static bool equal_(const CalibHistograms & first, const CalibHistograms & second)
    {
        return true
            and (equal(first.m_data, second.m_data))
        ;
    }
};

template <>
struct access<Pedestals>
{
    static bool equal_(const Pedestals & first, const Pedestals & second)
    {
        return true
            and (equal(first.m_pedestals, second.m_pedestals))
        ;
    }
};

template <>
struct access<Pedestals::Item>
{
    static bool equal_(const Pedestals::Item & first, const Pedestals::Item & second)
    {
        return true
            and (equal(first.m_mean, second.m_mean))
            and (equal(first.m_variance, second.m_variance))
        ;
    }
};

template <>
struct access<big>
{
    static bool equal_(const big & first, const big & second)
    {
        return true
            and (equal(first.tVector_, second.tVector_))
            and (equal(first.thVector_, second.thVector_))
            and (equal(first.sVector_, second.sVector_))
            and (equal(first.id_current, second.id_current))
            and (equal(first.index_id, second.index_id))
            and (equal(first.cota_current, second.cota_current))
            and (equal(first.cotb_current, second.cotb_current))
            and (equal(first.abs_cotb, second.abs_cotb))
            and (equal(first.fpix_current, second.fpix_current))
        ;
    }
};

template <>
struct access<big::bigEntry>
{
    static bool equal_(const big::bigEntry & first, const big::bigEntry & second)
    {
        return true
            and (equal(first.runnum, second.runnum))
            and (equal(first.alpha, second.alpha))
            and (equal(first.cotalpha, second.cotalpha))
            and (equal(first.beta, second.beta))
            and (equal(first.cotbeta, second.cotbeta))
            and (equal(first.costrk, second.costrk))
            and (equal(first.qavg, second.qavg))
            and (equal(first.symax, second.symax))
            and (equal(first.dyone, second.dyone))
            and (equal(first.syone, second.syone))
            and (equal(first.sxmax, second.sxmax))
            and (equal(first.dxone, second.dxone))
            and (equal(first.sxone, second.sxone))
            and (equal(first.dytwo, second.dytwo))
            and (equal(first.sytwo, second.sytwo))
            and (equal(first.dxtwo, second.dxtwo))
            and (equal(first.sxtwo, second.sxtwo))
            and (equal(first.qmin, second.qmin))
            and (equal(first.par, second.par))
            and (equal(first.ytemp, second.ytemp))
            and (equal(first.xtemp, second.xtemp))
            and (equal(first.avg, second.avg))
            and (equal(first.aqfl, second.aqfl))
            and (equal(first.chi2, second.chi2))
            and (equal(first.spare, second.spare))
        ;
    }
};

template <>
struct access<big::bigHeader>
{
    static bool equal_(const big::bigHeader & first, const big::bigHeader & second)
    {
        return true
            and (equal(first.title, second.title))
            and (equal(first.ID, second.ID))
            and (equal(first.NBy, second.NBy))
            and (equal(first.NByx, second.NByx))
            and (equal(first.NBxx, second.NBxx))
            and (equal(first.NFy, second.NFy))
            and (equal(first.NFyx, second.NFyx))
            and (equal(first.NFxx, second.NFxx))
            and (equal(first.vbias, second.vbias))
            and (equal(first.temperature, second.temperature))
            and (equal(first.fluence, second.fluence))
            and (equal(first.qscale, second.qscale))
            and (equal(first.s50, second.s50))
            and (equal(first.templ_version, second.templ_version))
        ;
    }
};

template <>
struct access<big::bigStore>
{
    static bool equal_(const big::bigStore & first, const big::bigStore & second)
    {
        return true
            and (equal(first.head, second.head))
            and (equal(first.entby, second.entby))
            and (equal(first.entbx, second.entbx))
            and (equal(first.entfy, second.entfy))
            and (equal(first.entfx, second.entfx))
        ;
    }
};

template <>
struct access<boostTypeObj>
{
    static bool equal_(const boostTypeObj & first, const boostTypeObj & second)
    {
        return true
            and (equal(first.a, second.a))
            and (equal(first.b, second.b))
            and (equal(first.aa, second.aa))
            and (equal(first.bb, second.bb))
        ;
    }
};

template <>
struct access<child>
{
    static bool equal_(const child & first, const child & second)
    {
        return true
            and (equal(static_cast<const mybase &>(first), static_cast<const mybase &>(second)))
            and (equal(first.b, second.b))
        ;
    }
};

template <>
struct access<condex::ConfF>
{
    static bool equal_(const condex::ConfF & first, const condex::ConfF & second)
    {
        return true
            and (equal(static_cast<const cond::BaseKeyed &>(first), static_cast<const cond::BaseKeyed &>(second)))
            and (equal(first.v, second.v))
            and (equal(first.key, second.key))
        ;
    }
};

template <>
struct access<condex::ConfI>
{
    static bool equal_(const condex::ConfI & first, const condex::ConfI & second)
    {
        return true
            and (equal(static_cast<const cond::BaseKeyed &>(first), static_cast<const cond::BaseKeyed &>(second)))
            and (equal(first.v, second.v))
            and (equal(first.key, second.key))
        ;
    }
};

template <>
struct access<condex::Efficiency>
{
    static bool equal_(const condex::Efficiency & first, const condex::Efficiency & second)
    {
        return true
        ;
    }
};

template <>
struct access<condex::ParametricEfficiencyInEta>
{
    static bool equal_(const condex::ParametricEfficiencyInEta & first, const condex::ParametricEfficiencyInEta & second)
    {
        return true
            and (equal(static_cast<const condex::Efficiency &>(first), static_cast<const condex::Efficiency &>(second)))
            and (equal(first.cutLow, second.cutLow))
            and (equal(first.cutHigh, second.cutHigh))
            and (equal(first.low, second.low))
            and (equal(first.high, second.high))
        ;
    }
};

template <>
struct access<condex::ParametricEfficiencyInPt>
{
    static bool equal_(const condex::ParametricEfficiencyInPt & first, const condex::ParametricEfficiencyInPt & second)
    {
        return true
            and (equal(static_cast<const condex::Efficiency &>(first), static_cast<const condex::Efficiency &>(second)))
            and (equal(first.cutLow, second.cutLow))
            and (equal(first.cutHigh, second.cutHigh))
            and (equal(first.low, second.low))
            and (equal(first.high, second.high))
        ;
    }
};

template <>
struct access<fakeMenu>
{
    static bool equal_(const fakeMenu & first, const fakeMenu & second)
    {
        return true
            and (equal(first.m_algorithmMap, second.m_algorithmMap))
        ;
    }
};

template <typename T, unsigned int S>
struct access<fixedArray<T, S>>
{
    static bool equal_(const fixedArray<T, S> & first, const fixedArray<T, S> & second)
    {
        return true
            and (equal(first.content, second.content))
        ;
    }
};

template <>
struct access<mySiStripNoises>
{
    static bool equal_(const mySiStripNoises & first, const mySiStripNoises & second)
    {
        return true
            and (equal(first.v_noises, second.v_noises))
            and (equal(first.indexes, second.indexes))
        ;
    }
};

template <>
struct access<mySiStripNoises::DetRegistry>
{
    static bool equal_(const mySiStripNoises::DetRegistry & first, const mySiStripNoises::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
            and (equal(first.iend, second.iend))
        ;
    }
};

template <>
struct access<mybase>
{
    static bool equal_(const mybase & first, const mybase & second)
    {
        return true
        ;
    }
};

template <>
struct access<mypt>
{
    static bool equal_(const mypt & first, const mypt & second)
    {
        return true
            and (equal(first.pt, second.pt))
        ;
    }
};

template <>
struct access<strKeyMap>
{
    static bool equal_(const strKeyMap & first, const strKeyMap & second)
    {
        return true
            and (equal(first.m_content, second.m_content))
        ;
    }
};

}
}

#endif

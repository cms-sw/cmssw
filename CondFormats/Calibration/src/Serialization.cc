
#include "CondFormats/Calibration/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void Algo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(a);
}
COND_SERIALIZATION_INSTANTIATE(Algo);

template <class Archive>
void AlgoMap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("std::map<std::string, Algo>", boost::serialization::base_object<std::map<std::string, Algo>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(AlgoMap);

template <class Archive>
void Algob::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(b);
}
COND_SERIALIZATION_INSTANTIATE(Algob);

template <class Archive>
void BlobComplex::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(objects);
}
COND_SERIALIZATION_INSTANTIATE(BlobComplex);

template <class Archive>
void BlobComplexContent::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data1);
    ar & BOOST_SERIALIZATION_NVP(data2);
    ar & BOOST_SERIALIZATION_NVP(data3);
}
COND_SERIALIZATION_INSTANTIATE(BlobComplexContent);

template <class Archive>
void BlobComplexData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(a);
    ar & BOOST_SERIALIZATION_NVP(b);
    ar & BOOST_SERIALIZATION_NVP(values);
}
COND_SERIALIZATION_INSTANTIATE(BlobComplexData);

template <class Archive>
void BlobComplexObjects::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(a);
    ar & BOOST_SERIALIZATION_NVP(b);
    ar & BOOST_SERIALIZATION_NVP(content);
}
COND_SERIALIZATION_INSTANTIATE(BlobComplexObjects);

template <class Archive>
void BlobNoises::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_noises);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}
COND_SERIALIZATION_INSTANTIATE(BlobNoises);

template <class Archive>
void BlobNoises::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}
COND_SERIALIZATION_INSTANTIATE(BlobNoises::DetRegistry);

template <class Archive>
void BlobPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_pedestals);
}
COND_SERIALIZATION_INSTANTIATE(BlobPedestals);

template <class Archive>
void CalibHistogram::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_histo);
}
COND_SERIALIZATION_INSTANTIATE(CalibHistogram);

template <class Archive>
void CalibHistograms::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_data);
}
COND_SERIALIZATION_INSTANTIATE(CalibHistograms);

template <class Archive>
void Pedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_pedestals);
}
COND_SERIALIZATION_INSTANTIATE(Pedestals);

template <class Archive>
void Pedestals::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_mean);
    ar & BOOST_SERIALIZATION_NVP(m_variance);
}
COND_SERIALIZATION_INSTANTIATE(Pedestals::Item);

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
COND_SERIALIZATION_INSTANTIATE(big);

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
COND_SERIALIZATION_INSTANTIATE(big::bigEntry);

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
COND_SERIALIZATION_INSTANTIATE(big::bigHeader);

template <class Archive>
void big::bigStore::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(head);
    ar & BOOST_SERIALIZATION_NVP(entby);
    ar & BOOST_SERIALIZATION_NVP(entbx);
    ar & BOOST_SERIALIZATION_NVP(entfy);
    ar & BOOST_SERIALIZATION_NVP(entfx);
}
COND_SERIALIZATION_INSTANTIATE(big::bigStore);

template <class Archive>
void boostTypeObj::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(a);
    ar & BOOST_SERIALIZATION_NVP(b);
    ar & BOOST_SERIALIZATION_NVP(aa);
    ar & BOOST_SERIALIZATION_NVP(bb);
}
COND_SERIALIZATION_INSTANTIATE(boostTypeObj);

template <class Archive>
void child::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("mybase", boost::serialization::base_object<mybase>(*this));
    ar & BOOST_SERIALIZATION_NVP(b);
}
COND_SERIALIZATION_INSTANTIATE(child);

template <class Archive>
void condex::ConfF::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::BaseKeyed", boost::serialization::base_object<cond::BaseKeyed>(*this));
    ar & BOOST_SERIALIZATION_NVP(v);
    ar & BOOST_SERIALIZATION_NVP(key);
}
COND_SERIALIZATION_INSTANTIATE(condex::ConfF);

template <class Archive>
void condex::ConfI::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::BaseKeyed", boost::serialization::base_object<cond::BaseKeyed>(*this));
    ar & BOOST_SERIALIZATION_NVP(v);
    ar & BOOST_SERIALIZATION_NVP(key);
}
COND_SERIALIZATION_INSTANTIATE(condex::ConfI);

template <class Archive>
void condex::Efficiency::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(condex::Efficiency);

template <class Archive>
void condex::ParametricEfficiencyInEta::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("condex::Efficiency", boost::serialization::base_object<condex::Efficiency>(*this));
    ar & BOOST_SERIALIZATION_NVP(cutLow);
    ar & BOOST_SERIALIZATION_NVP(cutHigh);
    ar & BOOST_SERIALIZATION_NVP(low);
    ar & BOOST_SERIALIZATION_NVP(high);
}
COND_SERIALIZATION_INSTANTIATE(condex::ParametricEfficiencyInEta);

template <class Archive>
void condex::ParametricEfficiencyInPt::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("condex::Efficiency", boost::serialization::base_object<condex::Efficiency>(*this));
    ar & BOOST_SERIALIZATION_NVP(cutLow);
    ar & BOOST_SERIALIZATION_NVP(cutHigh);
    ar & BOOST_SERIALIZATION_NVP(low);
    ar & BOOST_SERIALIZATION_NVP(high);
}
COND_SERIALIZATION_INSTANTIATE(condex::ParametricEfficiencyInPt);

template <class Archive>
void fakeMenu::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_algorithmMap);
}
COND_SERIALIZATION_INSTANTIATE(fakeMenu);

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
COND_SERIALIZATION_INSTANTIATE(mySiStripNoises);

template <class Archive>
void mySiStripNoises::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}
COND_SERIALIZATION_INSTANTIATE(mySiStripNoises::DetRegistry);

template <class Archive>
void mybase::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(mybase);

template <class Archive>
void mypt::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pt);
}
COND_SERIALIZATION_INSTANTIATE(mypt);

template <class Archive>
void strKeyMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_content);
}
COND_SERIALIZATION_INSTANTIATE(strKeyMap);

#include "CondFormats/Calibration/src/SerializationManual.h"

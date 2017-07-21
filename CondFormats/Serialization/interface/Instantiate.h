#pragma once

#include "CondFormats/Serialization/interface/Archive.h"

#include <boost/serialization/export.hpp>

// Instantiate serialization code. It works with template
// arguments as well (use one for each specialization)
#define COND_SERIALIZATION_INSTANTIATE(...) \
    template void __VA_ARGS__::serialize<cond::serialization::InputArchive    >(cond::serialization::InputArchive     & ar, const unsigned int); \
    template void __VA_ARGS__::serialize<cond::serialization::OutputArchive   >(cond::serialization::OutputArchive    & ar, const unsigned int); \
    template void __VA_ARGS__::serialize<cond::serialization::InputArchiveXML >(cond::serialization::InputArchiveXML  & ar, const unsigned int); \
    template void __VA_ARGS__::serialize<cond::serialization::OutputArchiveXML>(cond::serialization::OutputArchiveXML & ar, const unsigned int);

// Polymorphic classes must be registered as such
#define COND_SERIALIZATION_REGISTER_POLYMORPHIC(T) \
    BOOST_CLASS_EXPORT_IMPLEMENT(T);


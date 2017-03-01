#pragma once

// Missing in EOS' portable archive
#include <cassert>

#include "boost/archive/xml_iarchive.hpp"
#include "boost/archive/xml_oarchive.hpp"
#include "boost/archive/xml_oarchive.hpp"

#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"

namespace cond {
namespace serialization {

  typedef eos::portable_iarchive InputArchive;
  typedef eos::portable_oarchive OutputArchive;

  typedef boost::archive::xml_iarchive InputArchiveXML;
  typedef boost::archive::xml_oarchive OutputArchiveXML;

}
}


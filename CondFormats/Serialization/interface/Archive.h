#ifndef CondFormats_Serialization_Archive_H
#define CondFormats_Serialization_Archive_H

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace cond {
namespace serialization {

  typedef boost::archive::binary_iarchive InputArchive;
  typedef boost::archive::binary_oarchive OutputArchive;

}
}

#endif

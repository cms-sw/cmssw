#ifndef CondFormats_External_PIXELFEDCHANNEL_H
#define CondFormats_External_PIXELFEDCHANNEL_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>

#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"

// struct PixelFEDChannel {
//   unsigned int fed, link, roc_first, roc_last;
// };

namespace boost {
  namespace serialization {

    /*
     * Note regarding object tracking: all autos used here
     * must resolve to untracked types, since we use local
     * variables in the stack which could end up with the same
     * address. For the moment, all types resolved by auto here
     * are primitive types, which are untracked by default
     * by Boost Serialization.
     */

    template <class Archive>
    void save(Archive& ar, const PixelFEDChannel& obj, const unsigned int) {
      auto fed = obj.fed;
      auto link = obj.link;
      auto roc_first = obj.roc_first;
      auto roc_last = obj.roc_last;
      ar& boost::serialization::make_nvp("fed_", fed);
      ar& boost::serialization::make_nvp("link_", link);
      ar& boost::serialization::make_nvp("roc_first_", roc_first);
      ar& boost::serialization::make_nvp("roc_last_", roc_last);
    }

    template <class Archive>
    void load(Archive& ar, PixelFEDChannel& obj, const unsigned int) {
      unsigned int fed_;
      unsigned int link_;
      unsigned int roc_first_;
      unsigned int roc_last_;

      ar& boost::serialization::make_nvp("fed_", fed_);
      ar& boost::serialization::make_nvp("link_", link_);
      ar& boost::serialization::make_nvp("roc_first_", roc_first_);
      ar& boost::serialization::make_nvp("roc_last_", roc_last_);
      PixelFEDChannel tmp{fed_, link_, roc_first_, roc_last_};
      obj = tmp;
    }

    template <class Archive>
    void serialize(Archive& ar, PixelFEDChannel& obj, const unsigned int v) {
      split_free(ar, obj, v);
    }

  }  // namespace serialization
}  // namespace boost

#endif

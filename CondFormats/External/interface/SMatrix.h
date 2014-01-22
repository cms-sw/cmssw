#ifndef CondFormats_External_SMATRIX_H
#define CondFormats_External_SMATRIX_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

// std::vector used in DataFormats/EcalDetId/interface/EcalContainer.h
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>

#include <Math/SMatrix.h>

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


// Math/SMatrix.h                                                                                                                                                                                          
template<class Archive, typename T, unsigned int D1, unsigned int D2, class R>
void serialize(Archive & ar, ROOT::Math::SMatrix<T, D1, D2, R> & obj, const unsigned int)
{
  unsigned int i = 0;
  for (auto & value : obj) {
    ar & boost::serialization::make_nvp(std::to_string(i).c_str(), value);
    ++i;
  }
}

} // namespace serialization
} // namespace boost

namespace cond {
namespace serialization {

// Math/SMatrix.h
template <typename T, unsigned int D1, unsigned int D2, class R>
struct access<ROOT::Math::SMatrix<T, D1, D2, R>>
{
    static bool equal_(const ROOT::Math::SMatrix<T, D1, D2, R> & first, const ROOT::Math::SMatrix<T, D1, D2, R> & second)
    {
        return std::equal(first.begin(), first.end(), second.begin(),
            [](decltype(*first.begin()) a, decltype(a) b) -> bool {
                return equal(a, b);
            }
        );
    }
};

} // namespace serialization
} // namespace cond

#endif

#ifndef CondFormats_External_FFTJET_H
#define CondFormats_External_FFTJET_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

// std::vector used in DataFormats/EcalDetId/interface/EcalContainer.h
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequence.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetDict.h"

#include <stdexcept>

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

    // JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequence.h
    template <class Archive, class Jet, template <class> class InitialConverter, template <class> class FinalConverter>
    void serialize(Archive& ar,
                   FFTJetCorrectorSequence<Jet, InitialConverter, FinalConverter>& obj,
                   const unsigned int) {
      throw std::runtime_error("Unimplemented serialization code.");
    }

    // JetMETCorrections/FFTJetObjects/interface/FFTJetDict.h
    template <class Archive, class Key, class T, class Compare, class Allocator>
    void serialize(Archive& ar, FFTJetDict<Key, T, Compare, Allocator>& obj, const unsigned int) {
      throw std::runtime_error("Unimplemented serialization code.");
    }

  }  // namespace serialization
}  // namespace boost

#endif

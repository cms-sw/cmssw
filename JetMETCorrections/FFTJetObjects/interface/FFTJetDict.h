#ifndef JetMETCorrections_FFTJetObjects_FFTJetDict_h
#define JetMETCorrections_FFTJetObjects_FFTJetDict_h

//
// This template provides a variation of std::map with
// subscripting operator which does not automatically
// insert default values
//
// I. Volobouev
// 08/03/2012

#include <map>
#include "FWCore/Utilities/interface/Exception.h"

template <class Key, class T, class Compare = std::less<Key>, class Allocator = std::allocator<std::pair<const Key, T> > >
struct FFTJetDict : public std::map<Key, T, Compare, Allocator> {
  inline T& operator[](const Key&);
  inline const T& operator[](const Key&) const;
};

template <class Key, class T, class Compare, class Allocator>
T& FFTJetDict<Key, T, Compare, Allocator>::operator[](const Key& key) {
  typename FFTJetDict<Key, T, Compare, Allocator>::const_iterator it = this->find(key);
  if (it == std::map<Key, T, Compare, Allocator>::end())
    throw cms::Exception("KeyNotFound") << "FFTJetDict: key \"" << key << "\" not found\n";
  return const_cast<T&>(it->second);
}

template <class Key, class T, class Compare, class Allocator>
const T& FFTJetDict<Key, T, Compare, Allocator>::operator[](const Key& key) const {
  typename FFTJetDict<Key, T, Compare, Allocator>::const_iterator it = this->find(key);
  if (it == std::map<Key, T, Compare, Allocator>::end())
    throw cms::Exception("KeyNotFound") << "FFTJetDict: key \"" << key << "\" not found\n";
  return it->second;
}
#endif  // JetMETCorrections_FFTJetObjects_FFTJetDict_h

#ifndef OPERATORS_H_
#define OPERATORS_H_
#include "TH1F.h"
#include <iostream>
#include <vector>
#include <map>

/**
 * Operators.h
 * Various operators and handy methods for std::pairs and std::maps
 * 
 * \author Jamie Ballin
 * \date May 2008
 */

/*
 * Pair-wise addition of std::pairs
 */
template <class T> std::pair<T, T> operator+(const std::pair<T, T>& one, const std::pair<T, T>& two) {
	std::pair<T, T> r(one.first + two.first, one.second + two.second);
	return r;
}

/*
 * Pair-wise subtraction or std::pairs
 */
template <class T> std::pair<T, T> operator-(const std::pair<T, T>& one, const std::pair<T, T>& two) {
	std::pair<T, T> r(one.first - two.first, one.second - two.second);
	return r;
}

/*
 * Streams std::pair contents to stream
 */
template <class T> std::ostream& operator<<(std::ostream& s, const std::pair<T, T>& aT) {
	s << "(" << aT.first << ", " << aT.second << ")";
	return s;
}

/*
 * Extracts the value set from a std::map
 */
template <class K, class V> void valueVector(const std::map<K, V>& extract, std::vector<V>& output) {
	for(typename std::map<K, V>::const_iterator cit = extract.begin(); cit != extract.end(); ++cit) {
		output.push_back((*cit).second);
	}
}

/*
 * Extracts the key set from a std::map
 */
template <class K, class V> void keyVector(const std::map<K, V>& extract, std::vector<K>& output) {
	for(typename std::map<K, V>::const_iterator cit = extract.begin(); cit != extract.end(); ++cit) {
		output.push_back((*cit).first);
	}
}

#endif /*OPERATORS_H_*/

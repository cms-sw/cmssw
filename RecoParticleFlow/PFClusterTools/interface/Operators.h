#ifndef OPERATORS_H_
#define OPERATORS_H_
#include "TH1F.h"
#include <iostream>

template <class T> std::pair<T, T> operator+(const std::pair<T, T>& one, const std::pair<T, T>& two) {
	std::pair<T, T> r(one.first + two.first, one.second + two.second);
	return r;
}

template <class T> std::pair<T, T> operator-(const std::pair<T, T>& one, const std::pair<T, T>& two) {
	std::pair<T, T> r(one.first - two.first, one.second - two.second);
	return r;
}

template <class T> std::ostream& operator<<(std::ostream& s, const std::pair<T, T>& aT) {
	s << "(" << aT.first << ", " << aT.second << ")";
	return s;
}

#endif /*OPERATORS_H_*/


#ifndef TOSTR_H
#define TOSTR_H
#include <sstream>
#include <iostream>
template <class T> std::string toString(const T& aT) {
	std::ostringstream oss;
	oss << aT;
	return oss.str();
}

template <class T> std::string toString(const std::pair<T, T>& aT) {
	std::ostringstream oss;
	oss << "(" << aT.first << ", " << aT.second << ")";
	return oss.str();
}


#endif /*TOSTR_H */


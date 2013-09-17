/*
 * Utils.h
 *
 *  Created on: 1 Oct 2012
 *      Author: secre
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>
#include <algorithm>

namespace jsoncollector {
class Utils {

public:
	/**
	 * Convenience method to convert a vector of doubles into a vector of strings
	 */
	static std::vector<std::string> vectorDoubleToString(
			std::vector<double> doubleVector);

	/**
	 * Convenience method to convert a vector of strings into a vector of doubles
	 */
	static std::vector<double> vectorStringToDouble(
			std::vector<std::string> stringVector);

	/**
	 * Returns true if the strings match exactly
	 */
	static bool matchExactly(std::string s1, std::string s2);

	/**
	 * Returns a string of the type host_pid
	 */
	static void getHostAndPID(std::string& sHPid);

	/**
	 * Parses a string into a vector of integers
	 */
	static void stringToIntArray(std::vector<int>& theVector,
			const std::string& theString);

	/**
	 * Converts an int array to string
	 */
	static void intArrayToString(std::vector<int>& theVector,
			std::string& theString);

	/**
	 * Bumps up the value at an index of the array, growing it as necessary
	 * The input vector has to be initialized!
	 */
	static void bumpIndex(std::vector<int>& theVector, unsigned int index);
};
}

#endif /* UTILS_H_ */

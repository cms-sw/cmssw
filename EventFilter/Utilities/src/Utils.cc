/*
 * Utils.cc
 *
 *  Created on: 1 Oct 2012
 *      Author: secre
 */
/*
#include "EventFilter/Utilities/interface/Utils.h"
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <algorithm>

using namespace jsoncollector;
*/
/* not used
vector<string> Utils::vectorDoubleToString(const vector<double> & doubleVector) {
	vector<string> strVector;
	stringstream ss;
	for (unsigned int i = 0; i < doubleVector.size(); i++) {
		ss << doubleVector.at(i);
		strVector.push_back(ss.str());
		ss.str("");
	}
	return strVector;
}

vector<double> Utils::vectorStringToDouble(const vector<string>& stringVector) {
	vector<double> dblVector;
	for (unsigned int i = 0; i < stringVector.size(); i++)
		dblVector.push_back(atof(stringVector.at(i).c_str()));
	return dblVector;
}

bool Utils::matchExactly(string s1, string s2) {
	if (s1.find(s2) == 0 && s2.find(s1) == 0)
		return true;
	return false;
}
*/
/*
void Utils::stringToIntArray(vector<int>& theArray, const string& theString) {
	// remove whitespace and []
	string mod;
	mod.resize(theString.size());
	std::remove_copy(theString.begin(), theString.end(), mod.begin(), ' ');
	string mod2 = mod.substr(1, mod.size() - 1);

	// parse
	std::stringstream ss(mod2);
	int i;
	while (ss >> i) {
		theArray.push_back(i);
		char peek = ss.peek();
		if (peek == ',')
			ss.ignore();
	}
}

void Utils::bumpIndex(vector<int>& theVector, unsigned int index) {
	// if starting with empty vector
	if (theVector.size() == 0) {
		theVector.push_back(0);
	}
	// bump index without growing
	if (theVector.size() > index) {
		theVector[index]++;
	} else {
		// create and grow another vector to the required index
		vector<int> biggerVector(theVector);
		for (unsigned int i = theVector.size() - 1; i < index; i++) {
			biggerVector.push_back(0);
		}
		// bump the last element of the grown vector
		biggerVector[biggerVector.size() - 1]++;
		theVector = biggerVector;
	}
}
*/

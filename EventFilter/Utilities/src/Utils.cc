/*
 * Utils.cc
 *
 *  Created on: 1 Oct 2012
 *      Author: secre
 */

#include "../interface/Utils.h"
#include <sstream>
#include <cstdlib>
#include <unistd.h>

using namespace jsoncollector;
using std::string;
using std::vector;
using std::stringstream;
using std::atof;

vector<string> Utils::vectorDoubleToString(vector<double> doubleVector) {
	vector<string> strVector;
	stringstream ss;
	for (unsigned int i = 0; i < doubleVector.size(); i++) {
		ss << doubleVector[i];
		strVector.push_back(ss.str());
		ss.str("");
	}
	return strVector;
}

vector<double> Utils::vectorStringToDouble(vector<string> stringVector) {
	vector<double> dblVector;
	for (unsigned int i = 0; i < stringVector.size(); i++)
		dblVector.push_back(atof(stringVector[i].c_str()));
	return dblVector;
}

bool Utils::matchExactly(string s1, string s2) {
	if (s1.find(s2) == 0 && s2.find(s1) == 0)
		return true;
	return false;
}

void Utils::getHostAndPID(string& sHPid) {
	stringstream hpid;
	int pid = (int) getpid();
	char hostname[128];
	gethostname(hostname, sizeof hostname);
	hpid << hostname << "_" << pid;
	sHPid = hpid.str();
}

void Utils::stringToIntArray(vector<int>& theArray, const string& theString) {
	// remove [] and whitespace
	string mod = theString;
	mod = mod.substr(1, mod.size() - 1);

	string mod2;
	mod2.resize(mod.size());
	std::remove_copy(mod.begin(), mod.end(), mod2.begin(), ' ');

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

void Utils::intArrayToString(vector<int>& theArray, string& theString) {
	stringstream ss;
	ss << "[";
	for (unsigned int i = 0; i < theArray.size() - 1; i++) {
		ss << theArray[i];
		ss << ",";
	}
	if (theArray.size() > 0)
		ss << theArray[theArray.size() - 1] << "]";

	theString = ss.str();
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

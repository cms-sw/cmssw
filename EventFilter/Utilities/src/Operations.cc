/*
 * Operations.cc
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#include "../interface/Operations.h"
#include "../interface/Utils.h"
#include <sstream>

using namespace jsoncollector;
using std::vector;
using std::string;
using std::stringstream;

const string Operations::SUM = "sum";
const string Operations::AVG = "avg";
const string Operations::SAME = "same";
const string Operations::HISTO = "histo";
const string Operations::CAT = "cat";

double Operations::sum(vector<double> elems) {
	double added = elems.at(0);
	for (unsigned int i = 1; i < elems.size(); i++)
		added += elems.at(i);
	return added;
}

double Operations::avg(vector<double> elems) {
	return sum(elems) / elems.size();
}

string Operations::same(const vector<string>& elems) {
	for (unsigned int i = 0; i < elems.size() - 1; i++)
		if (!Utils::matchExactly(elems.at(i), elems.at(i + 1)) || elems.at(i).length()
				== 0)
			return "N/A";
	return elems.at(0);
}

string Operations::histo(const vector<string>& elems) {
	vector<vector<int> > inputHistos;

	for (unsigned int i = 0; i < elems.size(); i++) {
		vector<int> currentHisto;
		string currentHistoAsString = elems.at(i);
		Utils::stringToIntArray(currentHisto, currentHistoAsString);
		inputHistos.push_back(currentHisto);
	}

	if (inputHistos.size() > 0) {
		// initialize resulting histo to largest size
		unsigned int maxSize = 0;
		for (unsigned int i = 0; i < inputHistos.size(); i++) {
			if (inputHistos[i].size() > maxSize)
				maxSize = inputHistos[i].size();
		}

		// initialize the resulting histo to the size
		vector<int> resultingHisto(maxSize);
		for (unsigned int i = 0; i < resultingHisto.size(); i++)
			resultingHisto[i] = 0;

		for (unsigned int i = 0; i < inputHistos.size(); i++) {
			vector<int> currentHisto = inputHistos[i];
			for (unsigned int j = 0; j < currentHisto.size(); j++) {
				resultingHisto[j] += currentHisto[j];
			}
		}

		string resHistoAsString;
		Utils::intArrayToString(resultingHisto, resHistoAsString);
		return resHistoAsString;

	} else
		return "Cannot load input histos";
}

string Operations::cat(const vector<string>& elems) {
	stringstream ss;
	for (unsigned int i = 0; i < elems.size(); i++) {
		ss << elems.at(i);
		if (i != elems.size() - 1) {
			ss << ", ";
		}
	}
	return ss.str();
}

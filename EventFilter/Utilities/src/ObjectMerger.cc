/*
 * ObjectMerger.cc
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#include "../interface/ObjectMerger.h"
#include "../interface/Operations.h"
#include "../interface/Utils.h"
#include "../interface/FileIO.h"
#include "../interface/JSONSerializer.h"
#include <sstream>
#include <iostream>

using namespace jsoncollector;
using std::vector;
using std::string;
using std::stringstream;
using std::cout;
using std::endl;

DataPoint* ObjectMerger::merge(vector<DataPoint*> objectsToMerge,
		string& outcomeMessage, bool onlyHistos) {
	// vector of vectors containing all data in datapoints
	vector<vector<string> > mergedData;

	// check consistency of input files
	if (!checkConsistency(objectsToMerge, outcomeMessage))
		return NULL;

	// 1. Get the definition of these data points
	DataPointDefinition dpd;
	getDataPointDefinitionFor(objectsToMerge[0]->getDefinition(), dpd);

	// 1.1 Check if definition has exact no of elements specified
	if (objectsToMerge[0]->getData().size() != dpd.getLegend().size()) {
		outcomeMessage
				= "JSON files and DEFINITION do not have the same number of elements in vectors!";
		//delete dpd;
		return NULL;
	}

	// 2. Assemble merged vector
	for (unsigned int nMetric = 0; nMetric
			< objectsToMerge[0]->getData().size(); nMetric++) {
		// 2.1. assemble vector of n-th data elements
		vector<string> metricVector;
		for (unsigned int nObj = 0; nObj < objectsToMerge.size(); nObj++)
			metricVector.push_back(objectsToMerge[nObj]->getData()[nMetric]);
		mergedData.push_back(metricVector);
	}

	// 3. Apply defined operation for each measurement in all data points
	vector<string> outputValues;
	for (unsigned int i = 0; i < mergedData.size(); i++) {
		string strVal = "";
		if (onlyHistos) {
			if (Utils::matchExactly(dpd.getLegendFor(i).getOperation(),
					Operations::HISTO)) {
				strVal = applyOperation(mergedData[i],
						dpd.getLegendFor(i).getOperation());
			} else {
				strVal = mergedData[i][mergedData[i].size() - 1];
			}

		} else {
			strVal = applyOperation(mergedData[i],
					dpd.getLegendFor(i).getOperation());

		}
		outputValues.push_back(strVal);
	}

	// 4. Assemble output DataPoint
	/*
	 * Change @ 19.06.2013
	 * Merging no longer concatenates DataPoint sources.
	 * The merged object will contain as source all contributing files,
	 * added by the caller class.
	 *
	 */
	/*
	 stringstream ss;
	 for (unsigned int i = 0; i < objectsToMerge.size(); i++) {
	 ss << objectsToMerge[i]->getSource();
	 if (i != objectsToMerge.size(objectsToMerge) - 1)
	 ss << ", ";
	 }
	 string source = ss.str();
	 */

	string source = "";
	string definition = objectsToMerge[0]->getDefinition();
	DataPoint* outputDP = new DataPoint(source, definition, outputValues);

	// delete dpd after no longer needed
	//delete dpd;

	return outputDP;
}

bool ObjectMerger::getDataPointDefinitionFor(string defFilePath,
		DataPointDefinition& dpd) {
	string dpdString;
	bool readOK = FileIO::readStringFromFile(defFilePath, dpdString);
	// data point definition is bad!
	if (!readOK) {
		cout << "Cannot read from JSON definition path: " << defFilePath
				<< endl;
		return false;
	}
	JSONSerializer::deserialize(&dpd, dpdString);
	return true;
}

DataPoint* ObjectMerger::csvToJson(string& olCSV, DataPointDefinition* dpd,
		string defPath) {

	DataPoint* dp = new DataPoint();
	dp->setDefinition(defPath);

	vector<string> tokens;
	std::istringstream ss(olCSV);
	while (!ss.eof()) {
		string field;
		getline(ss, field, ',');
		tokens.push_back(field);
	}

	dp->resetData();

	for (unsigned int i = 0; i < tokens.size(); i++) {
		string currentOpName = dpd->getLegendFor(i).getOperation();
		int index = atoi(tokens[i].c_str());
		if (currentOpName.compare(Operations::HISTO) == 0) {
			vector<int> histo;
			Utils::bumpIndex(histo, index);
			string histoStr;
			Utils::intArrayToString(histo, histoStr);
			dp->addToData(histoStr);
		} else
			dp->addToData(tokens[i]);
	}

	return dp;
}

string ObjectMerger::applyOperation(std::vector<string> dataVector,
		std::string operationName) {
	string opResultString = "N/A";

	if (operationName.compare(Operations::SUM) == 0) {
		stringstream ss;
		double opResult = Operations::sum(
				Utils::vectorStringToDouble(dataVector));
		ss << opResult;
		opResultString = ss.str();

	} else if (operationName.compare(Operations::AVG) == 0) {
		stringstream ss;
		double opResult = Operations::avg(
				Utils::vectorStringToDouble(dataVector));
		ss << opResult;
		opResultString = ss.str();
	}
	/*
	 * ADD MORE OPERATIONS HERE
	 */
	else if (operationName.compare(Operations::SAME) == 0) {
		opResultString = Operations::same(dataVector);
	}

	else if (operationName.compare(Operations::HISTO) == 0) {
		opResultString = Operations::histo(dataVector);
	}

	else if (operationName.compare(Operations::CAT) == 0) {
			opResultString = Operations::cat(dataVector);
		}

	// OPERATION WAS NOT DEFINED
	else {
		cout << "Operation " << operationName << " is NOT DEFINED!" << endl;
	}

	return opResultString;
}

bool ObjectMerger::checkConsistency(std::vector<DataPoint*> objectsToMerge,
		std::string& outcomeMessage) {

	for (unsigned int i = 0; i < objectsToMerge.size() - 1; i++) {
		// 1. Check if all have the same definition
		if (objectsToMerge[i]->getDefinition().compare(
				objectsToMerge[i + 1]->getDefinition()) != 0) {
			outcomeMessage = "JSON files have inconsistent definitions!";
			return false;
		}
		// 2. Check if objects to merge have the same number of elements in data vector
		if (objectsToMerge[i]->getData().size()
				!= objectsToMerge[i + 1]->getData().size()) {
			outcomeMessage
					= "JSON files have inconsistent number of elements in the data vector!";
			return false;
		}
	}
	return true;
}

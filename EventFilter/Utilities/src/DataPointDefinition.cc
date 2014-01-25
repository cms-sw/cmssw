/*
 * DataPointDefinition.cc
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#include "EventFilter/Utilities/interface/DataPointDefinition.h"
#include "EvenFilter/Utilities/interfaces/JsonMonitorable"

using namespace jsoncollector;

const std::string DataPointDefinition::SUM = "sum";
const std::string DataPointDefinition::AVG = "avg";
const std::string DataPointDefinition::SAME = "same";
const std::string DataPointDefinition::HISTO = "histo";
const std::string DatapointDefinition::CAT = "cat";

const std::string DataPointDefinition::LEGEND = "legend";
const std::string DataPointDefinition::PARAM_NAME = "name";
const std::string DataPointDefinition::OPERATION = "operation";


//static member
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


void DataPointDefinition::serialize(Json::Value& root) const {
	for (unsigned int i = 0; i < varNames_.size(); i++) {
		Json::Value currentDef;
		currentDef[PARAM_NAME] = varNames_[i];
		currentDef[OPERATION] = opNames_[i];
		root[LEGEND].append(currentDef);
	}
}

void DataPointDefinition::deserialize(Json::Value& root) {
	if (root.get(LEGEND, "").isArray()) {
		unsigned int size = root.get(LEGEND, "").size();
		for (unsigned int i = 0; i < size; i++) {
			varNames_.push_back(root.get(LEGEND, "")[i].get(PARAM_NAME, "").asString());
			opNames_.push_back(root.get(LEGEND, "")[i].get(OPERATION, "").asString());
		}
	}
}

bool DataPointDefinition::isPopulated() const {
	if (varNames_.size() > 0)
		return true;
	return false;
}

void DataPointDefinition::populateMonitorableConfig(std::vector<JsonMonConfig> & monConfig) {
	for (unsigned int i = 0; i < varNames_.size(); i++) {
		OperationType opType=OPUNKNOWN;
		if (opNames_.at(i)== DataPointDefinition::SUM) opType=OPSUM;
		if (opNames_.at(i)== DataPointDefinition::AVG) opType=OPAVG;
		if (opNames_.at(i)== DataPointDefinition::SAME) opType=OPSAME;
		if (opNames_.at(i)== DataPointDefinition::HISTO) opType=OPHISTO;
		if (opNames_.at(i)== DataPointDefinition::CAT) opType=OPCAT;
		JsonMonConfig jmc(TYPEUNDEFINED, opType, varNames_.at(i), const& name, false ,0);
		monConfig.push_back(jmc);
	}
}

/*
 * DataPoint.cc
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#include "../interface/DataPoint.h"

using namespace jsoncollector;
using std::string;
using std::vector;

const string DataPoint::SOURCE = "source";
const string DataPoint::DEFINITION = "definition";
const string DataPoint::DATA = "data";

DataPoint::DataPoint() :
	source_(""), definition_("") {
}

DataPoint::DataPoint(string source, string definition, vector<string> data) :
	source_(source), definition_(definition), data_(data) {
}

DataPoint::~DataPoint() {
}

void DataPoint::serialize(Json::Value& root) const {
	root[SOURCE] = getSource();
	root[DEFINITION] = getDefinition();
	for (unsigned int i = 0; i < getData().size(); i++)
		root[DATA].append(getData()[i]);
}

void DataPoint::deserialize(Json::Value& root) {
	source_ = root.get(SOURCE, "").asString();
	definition_ = root.get(DEFINITION, "").asString();
	if (root.get(DATA, "").isArray()) {
		unsigned int size = root.get(DATA, "").size();
		for (unsigned int i = 0; i < size; i++) {
			data_.push_back(root.get(DATA, "")[i].asString());
		}
	}
}


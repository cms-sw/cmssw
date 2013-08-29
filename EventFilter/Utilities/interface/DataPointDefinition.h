/*
 * DataPointDefinition.h
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#ifndef DATAPOINTDEFINITION_H_
#define DATAPOINTDEFINITION_H_

#include <string>
#include <vector>
#include "JsonSerializable.h"
#include "LegendItem.h"

namespace jsoncollector {
class DataPointDefinition: public JsonSerializable {

public:
	DataPointDefinition();
	DataPointDefinition(std::vector<LegendItem> legend);
	virtual ~DataPointDefinition();

	/**
	 * JSON serialization procedure for this class
	 */
	virtual void serialize(Json::Value& root) const;
	/**
	 * JSON deserialization procedure for this class
	 */
	virtual void deserialize(Json::Value& root);
	/**
	 * Returns true if the legend_ has elements
	 */
	bool isPopulated() const;
	/**
	 * Returns a LegendItem object at the specified index
	 */
	LegendItem getLegendFor(unsigned int index) const;
	std::vector<LegendItem> getLegend() const {
		return legend_;
	}

	// JSON field names
	static const std::string LEGEND;
	static const std::string PARAM_NAME;
	static const std::string OPERATION;

private:
	std::vector<LegendItem> legend_;
};
}

#endif /* DATAPOINTDEFINITION_H_ */

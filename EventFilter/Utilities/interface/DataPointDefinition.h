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
//#include "JsonSerializable.h"

namespace jsoncollector {

class JsonMonConfig;
class JsonSerializable;

class DataPointDefinition: public JsonSerializable {

public:
	DataPointDefinition() {}

	//DataPointDefinition(const std::vector<std::string>& names, const std::vector<std::string>& operations);

	virtual ~DataPointDefinition() {}

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
	 * Returns a LegendItem object ref at the specified index
	 */
	std::vector<std::string> const& getNames(unsigned int index) {return varName_;}
	std::vector<std::string> const& getOperations(unsigned int index) {return opNames;}

	/**
	 * Loads a DataPointDefinition from a specified reference
	 */
	static bool getDataPointDefinitionFor(std::string defFilePath,
			DataPointDefinition& def);

	void populateMonConfig(std::vector<JsonMonConfig>& monConfig);

	//known JSON operation names
	static const std::string SUM;
	static const std::string AVG;
	static const std::string SAME;
	static const std::string HISTO;
	static const std::string CAT;

	// JSON field names
	static const std::string LEGEND;
	static const std::string PARAM_NAME;
	static const std::string OPERATION;

private:
	std::vector<std::string> varNames_;
	std::vector<std::string> opNames_;
};
}

#endif /* DATAPOINTDEFINITION_H_ */

/*
 * ObjectMerger.h
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#ifndef OBJECTMERGER_H_
#define OBJECTMERGER_H_

#include <vector>
#include "DataPoint.h"
#include "DataPointDefinition.h"

namespace jsoncollector {
class ObjectMerger {

public:
	/**
	 * Merges the DataPoint objects in the vector by getting their definition and applying the required operations
	 * If the onlyHistos arg is set to true, only histograms will be merged, will for other params the latest value
	 * (@ objectsToMerge.size() - 1) will be taken
	 */
	static DataPoint* merge(std::vector<DataPoint*> objectsToMerge,
			std::string& outcomeMessage, bool onlyHistos);

	/**
	 * Loads DataPointDefinition into the specified reference
	 */
	static bool getDataPointDefinitionFor(std::string defFilePath,
			DataPointDefinition& def);

	/**
	 * Transforms the CSV string into a DataPoint object using the definition
	 */
	static DataPoint* csvToJson(std::string& olCSV, DataPointDefinition* dpd,
			std::string defPath);

private:
	static std::string applyOperation(std::vector<std::string> dataVector,
			std::string operationName);
	static bool checkConsistency(std::vector<DataPoint*> objectsToMerge,
			std::string& outcomeMessage);

};
}

#endif /* OBJECTMERGER_H_ */

/*
 * DataPoint.h
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#ifndef DATAPOINT_H_
#define DATAPOINT_H_

#include <string>
#include <vector>
#include <memory>
#include <stdint.h>
#include <assert.h>
#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/JsonSerializable.h"


namespace jsoncollector {
class DataPoint: public JsonSerializable {

public:

	DataPoint() :{ }

	DataPoint(std::string source, std::string definition) :	source_(source), definition_(source) { }

	DataPoint(	std::vector<JsonMonitorable*> const& data, 
			std::vector<JsonMonConfigData> const& monConfig,
			unsigned int expectedUpdates = 1, unsigned int maxUpdates = 0);

	/**
	 * JSON serialization procedure for this class
	 */
	virtual void serialize(Json::Value& root) const;
	/**
	 * JSON deserialization procedure for this class
	 */
	virtual void deserialize(Json::Value& root);

	std::vector<std::string>& getData() const {
		return data_;
	}

	//static members for serialization of multiple DataPoints
        static void serialize( tbb::concurrent_vector<DataPoint*> & dataPoints, std::vector<JsonMonitorableConfig>& config);

	static std::string mergeAndSerializeMonitorables(tbb::concurrent_vector<DataPoint*> & dataPoints,
		std::vector<JsonMonitorableConfig>& config, unsigned int index);

	void snap();

	void resetAccumulators() {
		for (auto& i : dataNative_) i->resetValue();
		updates_=0;
	}

	unsigned int getUpdates() {return updates_;}

	JsonMonitorable *monitorableAt(unsigned int index) {
		if (monitored_.size()>index)
			return dataNative_[index].get();
		else return nullptr;
	}

	// JSON field names
	static const std::string SOURCE;
	static const std::string DEFINITION;
	static const std::string DATA;

protected:
	//old
	std::string definition_;
	std::string source_;
	std::vector<std::string> data_;

	std::vector<std::unique_ptr<JsonMonitorable>> dataNative_;
	const std::vector<JsonMonitorable*> *monitored_;
	unsigned int updates_;

};
}

#endif /* DATAPOINT_H_ */

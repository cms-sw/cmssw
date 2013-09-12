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
#include <stdint.h>
#include "JsonSerializable.h"

namespace jsoncollector {
class DataPoint: public JsonSerializable {

public:
	DataPoint();
	DataPoint(std::string source, std::string definition,
			std::vector<std::string> data);
	virtual ~DataPoint();

	/**
	 * JSON serialization procedure for this class
	 */
	virtual void serialize(Json::Value& root) const;
	/**
	 * JSON deserialization procedure for this class
	 */
	virtual void deserialize(Json::Value& root);

	std::string getSource() const {
		return source_;
	}
	std::string getDefinition() const {
		return definition_;
	}
	std::vector<std::string> getData() const {
		return data_;
	}

	void setSource(std::string source) {
		source_ = source;
	}
	void setDefinition(std::string definition) {
		definition_ = definition;
	}
	void addToData(std::string data) {
		data_.push_back(data);
	}
	void resetData() {
		data_.clear();
	}

	// JSON field names
	static const std::string SOURCE;
	static const std::string DEFINITION;
	static const std::string DATA;

protected:
	std::string source_, definition_;
	std::vector<std::string> data_;

};
}

#endif /* DATAPOINT_H_ */

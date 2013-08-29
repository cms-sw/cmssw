/*
 * JSONSerializer.cc
 *
 *  Created on: Aug 2, 2012
 *      Author: aspataru
 */

#include "../interface/JSONSerializer.h"

using namespace jsoncollector;
using std::string;

bool JSONSerializer::serialize(JsonSerializable* pObj, string& output) {
	if (pObj == NULL)
		return false;

	Json::Value serializeRoot;
	pObj->serialize(serializeRoot);

	Json::StyledWriter writer;
	output = writer.write(serializeRoot);

	return true;
}

bool JSONSerializer::deserialize(JsonSerializable* pObj, string& input) {
	if (pObj == NULL)
		return false;

	Json::Value deserializeRoot;
	Json::Reader reader;

	if (!reader.parse(input, deserializeRoot))
		return false;

	pObj->deserialize(deserializeRoot);

	return true;
}

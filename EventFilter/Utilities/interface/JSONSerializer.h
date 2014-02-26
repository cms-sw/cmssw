/*
 * JSONSerializer.h
 *
 *  Created on: Aug 2, 2012
 *      Author: aspataru
 */

#ifndef JSONSERIALIZER_H_
#define JSONSERIALIZER_H_

#include "EventFilter/Utilities/interface/JsonSerializable.h"

#include <string>

namespace jsoncollector {
class JSONSerializer {
public:
  JSONSerializer();
  virtual ~JSONSerializer();

  /**
   * Serializes a JsonSerializable object to output string
   */
  static bool serialize(JsonSerializable* pObj, std::string & output);
  /**
   * Deserializes input from a string to the JsonSerializable object
   */
  static bool deserialize(JsonSerializable* pObj, std::string & input);
};
}

#endif /* JSONSERIALIZER_H_ */

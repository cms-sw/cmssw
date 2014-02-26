/*
 * JsonSerializable.h
 *
 *  Created on: Aug 2, 2012
 *      Author: aspataru
 */

#ifndef JSONSERIALIZABLE_H_
#define JSONSERIALIZABLE_H_

#include "json.h"

namespace jsoncollector {
class JsonSerializable {
public:
  virtual ~JsonSerializable() {
  }
  ;
  virtual void serialize(Json::Value& root) const = 0;
  virtual void deserialize(Json::Value& root) = 0;
};
}

#endif /* JSONSERIALIZABLE_H_ */

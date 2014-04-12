/*
 * DataPointDefinition.cc
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#include "EventFilter/Utilities/interface/DataPointDefinition.h"
#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace jsoncollector;

const std::string DataPointDefinition::SUM = "sum";
const std::string DataPointDefinition::AVG = "avg";
const std::string DataPointDefinition::SAME = "same";
const std::string DataPointDefinition::HISTO = "histo";
const std::string DataPointDefinition::CAT = "cat";
const std::string DataPointDefinition::BINARYOR = "binaryOr";
const std::string DataPointDefinition::MERGE = "merge";

const std::string DataPointDefinition::LEGEND = "legend";
const std::string DataPointDefinition::DATA = "data";
const std::string DataPointDefinition::PARAM_NAME = "name";
const std::string DataPointDefinition::OPERATION = "operation";
const std::string DataPointDefinition::TYPE = "type";


//static member implementation
bool DataPointDefinition::getDataPointDefinitionFor(std::string& defFilePath, DataPointDefinition* dpd, const std::string *defaultGroup)
{
  std::string dpdString;
  bool readOK = FileIO::readStringFromFile(defFilePath, dpdString);
  // data point definition is missing!
  if (!readOK) {
    edm::LogWarning("DataPointDefinition") << "Cannot read from JSON definition path: " << defFilePath;
    return false;
  }
  if (!defaultGroup) dpd->setDefaultGroup(LEGEND);
  else dpd->setDefaultGroup(*defaultGroup);
  JSONSerializer::deserialize(dpd, dpdString);
  return true;
}

void DataPointDefinition::serialize(Json::Value& root) const
{
  for (unsigned int i = 0; i < varNames_.size(); i++) {
    Json::Value currentDef;
    currentDef[PARAM_NAME] = varNames_[i];
    currentDef[OPERATION] = opNames_[i];
    if (typeNames_[i].size()) //only if it was found
      currentDef[TYPE] = typeNames_[i];
    root[defaultGroup_].append(currentDef);
  }
}

void DataPointDefinition::deserialize(Json::Value& root)
{
  if (root.get(defaultGroup_, "").isArray()) {
    unsigned int size = root.get(defaultGroup_, "").size();
    for (unsigned int i = 0; i < size; i++) {
      varNames_.push_back(root.get(defaultGroup_, "")[i].get(PARAM_NAME, "").asString());
      opNames_.push_back(root.get(defaultGroup_, "")[i].get(OPERATION, "").asString());
      typeNames_.push_back(root.get(defaultGroup_, "")[i].get(TYPE, "").asString());
    }
  }
}

bool DataPointDefinition::isPopulated() const
{
  if (varNames_.size() > 0)
    return true;
  return false;
}


OperationType DataPointDefinition::getOperationFor(unsigned int index)
{
  OperationType opType=OPUNKNOWN;
  if (opNames_.at(index)== DataPointDefinition::SUM) opType=OPSUM;
  if (opNames_.at(index)== DataPointDefinition::AVG) opType=OPAVG;
  if (opNames_.at(index)== DataPointDefinition::SAME) opType=OPSAME;
  if (opNames_.at(index)== DataPointDefinition::HISTO) opType=OPHISTO;
  if (opNames_.at(index)== DataPointDefinition::CAT) opType=OPCAT;
  return opType;
}

void DataPointDefinition::addLegendItem(std::string const& name, std::string const& type, std::string const& operation)
{
  varNames_.push_back(name);
  typeNames_.push_back(type);
  opNames_.push_back(operation);
}

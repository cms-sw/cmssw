/*
 * RunInfoDBNames.cc
 *
 *  Created on: Feb 19, 2010
 *      Author: diguida
 */

#include "CondFormats/RunInfo/interface/RunInfoDBNames.h"

std::string RunInfoDBNames::BooleanTableName() {
	return std::string("RUNSESSION_BOOLEAN");
}
std::string RunInfoDBNames::DateTableName() {
	return std::string("RUNSESSION_DATE");
}
std::string RunInfoDBNames::DoubleTableName() {
	return std::string("RUNSESSION_DOUBLE");
}
std::string RunInfoDBNames::FloatTableName() {
	return std::string("RUNSESSION_FLOAT");
}
std::string RunInfoDBNames::IntegerTableName() {
	return std::string("RUNSESSION_INTEGER");
}
std::string RunInfoDBNames::ParameterTableName() {
	return std::string("RUNSESSION_PARAMETER");
}
std::string RunInfoDBNames::StringTableName() {
	return std::string("RUNSESSION_STRING");
}

std::string RunInfoDBNames::DataColumnNames::Value() {
  return std::string("VALUE");
}

std::string RunInfoDBNames::DataColumnNames::ValueIndex() {
  return std::string("VALUE_INDEX");
}

std::string RunInfoDBNames::DataColumnNames::RunSessionParameterId() {
  return std::string("RUNSESSION_PARAMETER_ID");
}

std::string RunInfoDBNames::DataColumnNames::Owner() {
  return std::string("OWNER");
}

std::string RunInfoDBNames::DataColumnNames::ValueId() {
  return std::string("VALUE_ID");
}

std::string RunInfoDBNames::DataColumnNames::ParentId() {
  return std::string("PARENT_ID");
}

std::string RunInfoDBNames::ParameterColumnNames::Id() {
  return std::string("ID");
}

std::string RunInfoDBNames::ParameterColumnNames::RunNumber() {
  return std::string("RUNNUMBER");
}

std::string RunInfoDBNames::ParameterColumnNames::SessionId() {
  return std::string("SESSION_ID");
}

std::string RunInfoDBNames::ParameterColumnNames::Name() {
  return std::string("NAME");
}

std::string RunInfoDBNames::ParameterColumnNames::StringValue() {
  return std::string("STRING_VALUE");
}

std::string RunInfoDBNames::ParameterColumnNames::Time() {
  return std::string("TIME");
}

std::string RunInfoDBNames::ParameterColumnNames::Owner() {
  return std::string("OWNER");
}

std::string RunInfoDBNames::RunSessionDelimiterViewNames::Name() {
	return std::string("NAME");
}

std::string RunInfoDBNames::RunSessionDelimiterViewNames::ValueBool() {
	return std::string("VALUE_BOOL");
}

std::string RunInfoDBNames::RunSessionDelimiterViewNames::ValueDouble() {
	return std::string("VALUE_DOUBLE");
}

std::string RunInfoDBNames::RunSessionDelimiterViewNames::ValueInt() {
	return std::string("VALUE_INT");
}

std::string RunInfoDBNames::RunSessionDelimiterViewNames::ValueString() {
	return std::string("VALUE_STRING");
}

std::string RunInfoDBNames::RunSessionDelimiterViewNames::ValueTime() {
	return std::string("VALUE_TIME");
}

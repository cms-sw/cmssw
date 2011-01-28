/*
 * RunInfoDBNames.h
 *
 *  Created on: Feb 19, 2010
 *      Author: diguida
 */

#ifndef CondTools_RunInfo_RunInfoDBNames_h
#define CondTools_RunInfo_RunInfoDBNames_h

#include <string>

class RunInfoDBNames {
 public:
	static std::string BooleanTableName();
	static std::string DateTableName();
	static std::string DoubleTableName();
	static std::string FloatTableName();
	static std::string IntegerTableName();
	static std::string ParameterTableName();
	static std::string StringTableName();
        struct DataColumnNames {
	        static std::string Value();
	        static std::string ValueIndex();
	        static std::string RunSessionParameterId();
	        static std::string Owner();
	        static std::string ValueId();
	        static std::string ParentId();
	};
        struct ParameterColumnNames {
	        static std::string Id();
	        static std::string RunNumber();
	        static std::string SessionId();
	        static std::string Name();
	        static std::string StringValue();
	        static std::string Time();
	        static std::string Owner();
	};
        struct RunSessionDelimiterViewNames {
        	static std::string Name();
        	static std::string ValueString();
        	static std::string ValueBool();
        	static std::string ValueTime();
        	static std::string ValueDouble();
        	static std::string ValueInt();
        };
};

#endif /* CondTools_RunInfo_RunInfoDBNames_h */

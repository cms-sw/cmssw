#ifndef DBSTATE_H
#define DBSTATE_H

#include <string>
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"


namespace popcon
{


	struct DBInfo
	{
		std::string schema;
		std::string top_level_table;
		std::string object_name;
	};

	class DBState
	{
		public:
			DBState(){};
			DBState(coral::AttributeList);
			DBState(std::string,std::string,int);
			~DBState(){}
			void set_state(coral::AttributeList);

			coral::AttributeList update_helper(std::string& updateAction, std::string& updateCondition);


			//TODO maybe private - ?
		public:
			std::string name;
			std::string schema;
			int payload_size;
			std::string except_description;
			std::string manual_override;
	};

	bool operator==(const DBState& a, const DBState& b);


}


#endif

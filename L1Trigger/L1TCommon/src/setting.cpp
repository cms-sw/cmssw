#include <strstream>

#include "L1Trigger/L1TCommon/interface/setting.h"

namespace l1t{
	
setting::setting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole) : 
type_(type),
id_(id),
value_(value),
procRole_(procRole)
{
	if ( type.find("bool") != std::string::npos )
	{
		std::ostringstream convString;

		if ( type.find("vector") != std::string::npos )
		{
			std::string delim(","); //TODO: should be read dynamically
			std::vector<std::string> vals;
			if ( !parse ( value_.c_str(),
			(
				  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( vals ) ] % delim.c_str() )
			), boost::spirit::classic::nothing_p ).full )
			{  	
				throw std::runtime_error ("Wrong value format: " + value_);
			}

			for(std::vector<std::string>::iterator it=vals.begin(); it!=vals.end(); it++)
			{
				if ( it->find("true") != std::string::npos )
					convString << "1, ";
				else
					convString << "0, ";
			}
		}
		else
		{
			if ( value.find("true") != std::string::npos )
				convString << "1";
			else
				convString << "0";
		}

		value_ = convString.str();
	}
}

setting::setting(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim) :
type_("table"),
id_(id),
procRole_(procRole)
{
	str2VecStr_(columns, delim, tableColumns_);
	// if ( !parse ( columns.c_str(),
	// (
	// 	  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( tableColumns_ ) ] % delim.c_str() )
	// ), boost::spirit::classic::nothing_p ).full )
	// {  	
	// 	throw std::runtime_error ("Wrong value format: " + columns);
	// }
	str2VecStr_(types, delim, tableTypes_);

	// if ( !parse ( types.c_str(),
	// (
	// 	  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( tableTypes_ ) ] % delim.c_str() )
	// ), boost::spirit::classic::nothing_p ).full )
	// {  	
	// 	throw std::runtime_error ("Wrong value format: " + types);
	// }

	for (auto it=rows.begin(); it!=rows.end(); it++)
	{
		std::vector<std::string> aRow;
		str2VecStr_(*it, delim, aRow);
		// if ( !parse ( *it.c_str(),
		// (
		// 	  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( aRow ) ] % delim.c_str() )
		// ), boost::spirit::classic::nothing_p ).full )
		// {  	
		// 	throw std::runtime_error ("Wrong value format: " + *it);
		// }
		tableRow temp(aRow);
		temp.setRowTypes(tableTypes_);
		temp.setRowColumns(tableColumns_);
		tableRows_.push_back(temp);

	}
}

setting::~setting()
{
	;
}


l1t::LUT setting::getLUT(size_t addrWidth, size_t dataWidth, int padding, std::string delim)
{
	if ( type_.find("vector:uint") == std::string::npos )
		throw std::runtime_error("Cannot build LUT from type: " + type_ + ". Only vector:uint is allowed.");

	if ( delim.empty() )
		delim = ",";
	
	std::vector<unsigned int> vec = getVector<unsigned int>(delim);
	std::stringstream ss;
        ss << "#<header> V1 " << addrWidth << " " << dataWidth << " </header>" << std::endl;
        size_t i = 0;
	for (; i < vec.size() && i < (size_t)(1<<addrWidth); ++i) {
		ss << i << " " << vec[i] << std::endl;
	}
        // add padding to 2^addrWidth rows
        if (padding >= 0 && i < (size_t)(1<<addrWidth)) {
		for (; i < (size_t)(1<<addrWidth); ++i) {
			ss << i << " " << padding << std::endl;
		}
	}
	
	l1t::LUT lut;
	lut.read(ss);
	
	return lut;
}

setting& setting::operator=(const setting& aSet)
{
	value_ = aSet.value_;
	id_ = aSet.id_;
	type_ = aSet.type_;
	procRole_ = aSet.procRole_;
	return *this;
}

void setting::addTableRow(const std::string& row, std::string delim)
{
	if (type_.find("table") == std::string::npos)
		throw std::runtime_error("You cannot add a table row in type: " + type_ + ". Type is not table.");

	if ( delim.empty() )
		delim = std::string(",");
	std::vector<std::string> vals;
	str2VecStr_(row, delim, vals);
	// if ( !parse ( row.c_str(),
	// (
	// 	  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( vals ) ] % delim.c_str() )
	// ), boost::spirit::classic::nothing_p ).full )
	// {   	
	// 	throw std::runtime_error ("Wrong value format: " + row);
	// }
	tableRow tempRow(vals);
	tempRow.setRowTypes(tableTypes_);
	tempRow.setRowColumns(tableColumns_);
	tableRows_.push_back(tempRow);
}

void setting::setTableTypes(const std::string& types)
{	
	if (type_.find("table") == std::string::npos)
		throw std::runtime_error("You cannot set table types in type: " + type_ + ". Type is not table.");
	std::string delim(","); //TODO: should be read dynamically

	str2VecStr_(types, delim, tableTypes_);

	// if ( !parse ( std::string(types+delim).c_str(),
	// (
	// 	  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( tableTypes_ ) ] % delim.c_str() )
	// ), boost::spirit::classic::nothing_p ).full )
	// {  	
	// 	throw std::runtime_error ("Wrong value format: " + types);
	// }
}

void setting::setTableColumns(const std::string& cols)
{
	if (type_.find("table") == std::string::npos)
		throw std::runtime_error("You cannot set table columns in type: " + type_ + ". Type is not table.");
	std::string delim(","); //TODO: should be read dynamically

	str2VecStr_(cols, delim, tableColumns_);
	
	/*if ( !parse ( cols.c_str(),
	(
		  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( tableColumns_ ) ] % delim.c_str() )
	), boost::spirit::classic::nothing_p ).full )
	{  	
		throw std::runtime_error ("Wrong value format: " + cols);
	}*/
}

std::string tableRow::getRowAsStr()
{
	std::ostringstream str;
	for (auto it=row_.begin(); it!=row_.end(); it++)
		str << *it << " ";

	return str.str();
}

void setting::str2VecStr_(const std::string& aStr, const std::string& delim, std::vector<std::string>& aVec)
{
	if ( !parse ( aStr.c_str(),
	(
		  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( aVec ) ] % delim.c_str() )
	), boost::spirit::classic::nothing_p ).full )
	{  	
		throw std::runtime_error ("Wrong value format: " + aStr);
	}

	for(auto it = aVec.begin(); it != aVec.end(); it++) 
	{
		while (*(it->begin()) == ' ')
			it->erase(it->begin());
		while (*(it->end()-1) == ' ')
            it->erase(it->end()-1);
	}
}

}


#include <strstream>

#include "L1Trigger/L1TCommon/interface/setting.h"

namespace l1t{
	
setting::setting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole) : 
_type(type),
_id(id),
_value(value),
_procRole(procRole)
{
	if ( type.find("bool") != std::string::npos )
	{
		std::ostringstream convString;

		if ( type.find("vector") != std::string::npos )
		{
			std::string delim(","); //TODO: should be read dynamically
			std::vector<std::string> vals;
			if ( !parse ( std::string(erSp(_value, delim)+delim).c_str(),
			(
				  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( vals ) ] % delim.c_str() )
			), boost::spirit::classic::nothing_p ).full )
			{  	
				throw std::runtime_error ("Wrong value format: " + _value);
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

		_value = convString.str();
	}
}

setting::setting(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim) :
_type("table"),
_id(id),
_procRole(procRole)
{
	if ( !parse ( std::string(erSp(columns, delim)+delim).c_str(),
	(
		  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( _tableColumns ) ] % delim.c_str() )
	), boost::spirit::classic::nothing_p ).full )
	{  	
		throw std::runtime_error ("Wrong value format: " + columns);
	}

	if ( !parse ( std::string(erSp(types, delim)+delim).c_str(),
	(
		  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( _tableTypes ) ] % delim.c_str() )
	), boost::spirit::classic::nothing_p ).full )
	{  	
		throw std::runtime_error ("Wrong value format: " + types);
	}

	for (auto it=rows.begin(); it!=rows.end(); it++)
	{
		std::vector<std::string> aRow;
		if ( !parse ( std::string(erSp(*it, delim)+delim).c_str(),
		(
			  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( aRow ) ] % delim.c_str() )
		), boost::spirit::classic::nothing_p ).full )
		{  	
			throw std::runtime_error ("Wrong value format: " + *it);
		}
		tableRow temp(aRow);
		temp.setRowTypes(_tableTypes);
		temp.setRowColumns(_tableColumns);
		_tableRows.push_back(temp);

	}
}

setting::~setting()
{
	;
}


l1t::LUT setting::getLUT(size_t addrWidth, size_t dataWidth, int padding, std::string delim)
{
	if ( _type.find("vector:uint") == std::string::npos )
		throw std::runtime_error("Cannot build LUT from type: " + _type + ". Only vector:uint is allowed.");

	std::vector<unsigned int> vec = getVector<unsigned int>(delim);
	std::stringstream ss;
        ss << "#<header> V1 " << addrWidth << " " << dataWidth << " </header>" << std::endl;
        size_t i = 0;
	for (unsigned int i=0; i < vec.size() && i < (size_t)(1<<addrWidth); ++i)
		ss << i << " " << vec[i] << std::endl;

    // add padding to 2^addrWidth rows
    if (padding >= 0 && i < (size_t)(1<<addrWidth)) 
    {
		for (; i < (size_t)(1<<addrWidth); ++i)
			ss << i << " " << padding << std::endl;
	}
	
	l1t::LUT lut;
	lut.read(ss);
	
	return lut;
}

setting& setting::operator=(const setting& aSet)
{
	_value = aSet._value;
	_id = aSet._id;
	_type = aSet._type;
	_procRole = aSet._procRole;
	return *this;
}

void setting::addTableRow(const std::string& row, const std::string& delim)
{
	if (_type.find("table") == std::string::npos)
		throw std::runtime_error("You cannot add a table row in type: " + _type + ". Type is not table.");

	
	std::vector<std::string> vals;
	if ( !parse ( std::string(erSp(row, delim)+delim).c_str(),
	(
		  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( vals ) ] % delim.c_str() )
	), boost::spirit::classic::nothing_p ).full )
	{   	
		throw std::runtime_error ("Wrong value format: " + row);
	}
	tableRow tempRow(vals);
	tempRow.setRowTypes(_tableTypes);
	tempRow.setRowColumns(_tableColumns);
	_tableRows.push_back(tempRow);
}

void setting::setTableTypes(const std::string& types)
{	
	if (_type.find("table") == std::string::npos)
		throw std::runtime_error("You cannot set table types in type: " + _type + ". Type is not table.");
	std::string delim(","); //TODO: should be read dynamically

	if ( !parse ( std::string(types+delim).c_str(),
	(
		  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( _tableTypes ) ] % delim.c_str() )
	), boost::spirit::classic::nothing_p ).full )
	{  	
		throw std::runtime_error ("Wrong value format: " + types);
	}
}

void setting::setTableColumns(const std::string& cols)
{
	if (_type.find("table") == std::string::npos)
		throw std::runtime_error("You cannot set table columns in type: " + _type + ". Type is not table.");
	std::string delim(","); //TODO: should be read dynamically
	
	if ( !parse ( std::string(erSp(cols, delim)+delim).c_str(),
	(
		  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( _tableColumns ) ] % delim.c_str() )
	), boost::spirit::classic::nothing_p ).full )
	{  	
		throw std::runtime_error ("Wrong value format: " + cols);
	}
}

std::string tableRow::getRowAsStr()
{
	std::ostringstream str;
	for (auto it=_row.begin(); it!=_row.end(); it++)
		str << *it << " ";

	return str.str();
}

std::string setting::erSp(std::string str, const std::string& delim)
{
	if ( delim != " " )
		str.erase( std::remove_if( str.begin(), str.end(), ::isspace ), str.end() );
    
    return str;
}

}


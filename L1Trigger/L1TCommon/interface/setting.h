#ifndef __setting_h__
#define __setting_h__

#include <vector>
#include <string>

#include "CondFormats/L1TObjects/interface/LUT.h"

//boost libraries
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/classic.hpp>
#include <boost/spirit/include/classic_push_back_actor.hpp>

namespace l1t{

class tableRow
{
	public:
		tableRow() {};
		tableRow(const std::vector<std::string>& row) { _row = row;} ;
		void setRowTypes(const std::vector<std::string>& types) { _types = types; };
		void setRowColumns(const std::vector<std::string>& columns) { _columns = columns; };
		~tableRow() {};
		std::vector<std::string> getRow () { return _row; };
		template <class varType> varType getRowValue(const std::string& col);
	private:
		std::vector<std::string> _row;
		std::vector<std::string> _types;
		std::vector<std::string> _columns;
};


	
class setting
{
	public:
		setting() {};
		setting(std::string type, std::string id, std::string value, std::string procRole);
		void setProcRole(std::string procRole) { _procRole = procRole; };
		void setValue(std::string value) {_value = value; };
		void setId(std::string id) { _id = id; } ;
		void addTableRow(std::string row);
		void setTableTypes(std::string types);
		void setTableColumns(std::string cols);
		std::string getProcRole() { return _procRole; };
		std::string getValueAsStr() { return _value; };
		std::string getType() { return _type; };
		std::string getId() { return _id; } ;
		template <class varType> varType getValue();
		template <class varType> std::vector<varType> getVector(std::string delim = ",");
		std::vector<std::vector<std::string> > getTableRows();
		
		l1t::LUT getLUT(size_t addrWidth, size_t dataWidth, int padding = -1, std::string delim = ",");
		~setting();

		setting& operator=(const setting& aSet);
	private:
		std::string _type, _id, _value, _procRole;
		std::vector<tableRow> _tableRows;
		std::vector<std::string> _tableTypes;
		std::vector<std::string> _tableColumns;


};


template <typename varType> std::vector<varType> setting::getVector(std::string delim)
{
	if ( _type.find("vector") == std::string::npos )
		throw std::runtime_error("The registered type: " + _type + " is not vector so you need to call the getValue method");

	std::vector<varType> vals;

	if ( !parse ( std::string(_value+delim+" ").c_str(),
          (
          	* ( boost::spirit::classic::uint_p[boost::spirit::classic::push_back_a ( vals ) ] >> delim.c_str() >> *boost::spirit::classic::space_p )
          ), boost::spirit::classic::nothing_p ).full )
	{ 	
		throw std::runtime_error ("Wrong value format: " + _value);
	}

	return vals;
}

template <class varType> varType setting::getValue()
{
	if ( _type.find("vector") != std::string::npos )
		throw std::runtime_error("The registered type: " + _type + " is vector so you need to call the getVector method");
	
	return boost::lexical_cast<varType>(_value);
}

template <class varType> varType tableRow::getRowValue(const std::string& col)
{
	bool found(false);
	for (unsigned int i = 0; i < _columns.size(); i++)
	{
		if (_columns.at(i).find(col) != std::string::npos)
		{
			found = true;
			return boost::lexical_cast<varType>(_row.at(i));
		}
	}
	if (!found)
		throw std::runtime_error ("Column " + col + "not found.");
}






}
#endif


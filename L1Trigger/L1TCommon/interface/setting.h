#ifndef __setting_h__
#define __setting_h__

#include <vector>
#include <string>

#include "CondFormats/L1TObjects/interface/LUT.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

//boost libraries
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/classic.hpp>
#include <boost/spirit/include/classic_push_back_actor.hpp>

namespace l1t{

class tableRow
{
	public:
		tableRow() {};
		tableRow(const std::vector<std::string>& row) { row_ = row;} ;
		void setRowTypes(const std::vector<std::string>& types) { types_ = types; };
		void setRowColumns(const std::vector<std::string>& columns) { columns_ = columns; };
		~tableRow() {};
		std::vector<std::string> getRow () { return row_; };
		std::string getRowAsStr();
		template <class varType> varType getRowValue(const std::string& col);
	private:
		std::vector<std::string> row_;
		std::vector<std::string> types_;
		std::vector<std::string> columns_;

};


	
class setting
{
	public:
		setting() {};
		setting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole);
		setting(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim);
		void setProcRole(const std::string& procRole) { procRole_ = procRole; };
		void setValue(const std::string& value) {value_ = value; };
		void setId(const std::string& id) { id_ = id; } ;
		void addTableRow(const std::string& row, const std::string& delim=",");
		void resetTableRows() { tableRows_.clear();};
		void setTableTypes(const std::string& types);
		void setTableColumns(const std::string& cols);
		std::string getProcRole() { return procRole_; };
		std::string getValueAsStr() { return value_; };
		std::string getType() { return type_; };
		std::string getId() { return id_; } ;
		template <class varType> varType getValue();
		template <class varType> std::vector<varType> getVector(std::string delim = ",");
		std::vector<tableRow>  getTableRows() { return tableRows_; };
		l1t::LUT getLUT(size_t addrWidth, size_t dataWidth, int padding = -1, std::string delim = ",");
		~setting();

		setting& operator=(const setting& aSet);
	private:
		std::string type_, id_, value_, procRole_;
		std::vector<tableRow> tableRows_;
		std::vector<std::string> tableTypes_;
		std::vector<std::string> tableColumns_;
		
		std::string erSp_(std::string str, const std::string& delim);
};


template <typename varType> std::vector<varType> setting::getVector(std::string delim)
{
	if ( type_.find("vector") == std::string::npos )
		throw std::runtime_error("The registered type: " + type_ + " is not vector so you need to call the getValue method");

	std::vector<std::string> vals;
	if ( !parse ( std::string(erSp_(value_, delim)+delim).c_str(),
	(
		  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( vals ) ] % delim.c_str() )
	), boost::spirit::classic::nothing_p ).full )
	{  	
		throw std::runtime_error ("Wrong value format: " + value_);
	}
	vals.erase(vals.end()-1);

	std::vector<varType> newVals;
	for(auto it=vals.begin(); it!=vals.end(); it++)
		newVals.push_back(boost::lexical_cast<varType>(*it));

	edm::LogInfo ("l1t::setting::getVector") << "Returning vector with values " << this->getValueAsStr();
	return newVals;
}

template <class varType> varType setting::getValue()
{
	if ( type_.find("vector") != std::string::npos )
		throw std::runtime_error("The registered type: " + type_ + " is vector so you need to call the getVector method");
	
	edm::LogInfo ("l1t::setting::getValue") << "Returning value " << this->getValueAsStr();
	return boost::lexical_cast<varType>(value_);
}

template <class varType> varType tableRow::getRowValue(const std::string& col)
{
	bool found(false);
	int ct;
	for (unsigned int i = 0; i < columns_.size(); i++)
	{
		if (columns_.at(i).find(col) != std::string::npos)
		{
			found = true;
			ct = i;
		}
	}
	if (!found)
		throw std::runtime_error ("Column " + col + "not found.");

	edm::LogInfo ("l1t::setting::getRowValue") << "Returning value " << boost::lexical_cast<varType>(row_.at(ct)) <<  " from table row " << this->getRowAsStr();
	return boost::lexical_cast<varType>(row_.at(ct));
}






}
#endif


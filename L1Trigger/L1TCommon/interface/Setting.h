#ifndef L1Trigger_L1TCommon_Setting_h
#define L1Trigger_L1TCommon_Setting_h

#include <vector>
#include <string>
#include <memory>

#include "CondFormats/L1TObjects/interface/LUT.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "L1Trigger/L1TCommon/interface/Tools.h"

//boost libraries
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/classic.hpp>
#include <boost/spirit/include/classic_push_back_actor.hpp>

namespace l1t{

class TableRow
{
	public:
		TableRow() {};
		TableRow(const std::vector<std::string>& row, std::string *lt);
		void setTableId(const std::string& id) { tableId_ = id; };
		void setRowTypes(const std::vector<std::string>& types);
		void setRowColumns(const std::vector<std::string>& columns);
		void setLogStringVar(std::string* strVar) {logText_ = strVar;};
		~TableRow() {};
		std::vector<std::string> getRow () { return *row_; };
		std::string getRowAsStr();
		std::vector<std::string> getColumnNames() {return *columns_;};
		template <class varType> varType getRowValue(const std::string& col);
	private:
		std::string tableId_;
		std::string* logText_;
		//here we use the shared just because it reduces significantly the time
		std::shared_ptr< std::vector<std::string> > row_;
		std::shared_ptr< std::vector<std::string> > types_;
		std::shared_ptr< std::vector<std::string> > columns_;
		std::shared_ptr< std::map<std::string,int> > colDict_;

};


class Setting
{
	public:
		Setting() {};
		Setting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole, std::string *lt, const std::string& delim = "");
		Setting(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, std::string *lt, const std::string& delim);
		void setProcRole(const std::string& procRole) { procRole_ = procRole; };
		void setValue(const std::string& value);
		void setId(const std::string& id) { id_ = id; } ;
		void setLogStringVar(std::string* strVar) {logText_ = strVar;};
		void addTableRow(const std::string& row, const std::vector<std::string>& types, const std::vector<std::string>& columns);
		void resetTableRows() { TableRows_.clear();};
		std::string getProcRole() { return procRole_; };
		std::string getValueAsStr() { return value_; };
		std::string getType() { return type_; };
		std::string getId() { return id_; } ;
		template <class varType> varType getValue();
		template <class varType> std::vector<varType> getVector();
		std::vector<TableRow>  getTableRows() { return TableRows_; };
		l1t::LUT getLUT(size_t addrWidth = 0, size_t dataWidth = 31, int padding = -1, std::string delim = ","); // if the addrWidth parameter is 0 calculate the address width from the LUT length. 31 is the maximal supported number of bits for the output width of l1t::LUT
		~Setting();

		Setting& operator=(const Setting& aSet);
	private:
		std::string type_, id_, value_, procRole_, delim_;
		std::vector<TableRow> TableRows_;
		std::string* logText_;		
};


template <typename varType> std::vector<varType> Setting::getVector()
{
	
	if ( type_.find("vector") == std::string::npos )
		throw std::runtime_error("The registered type: " + type_ + " is not vector so you need to call the getValue method");

	std::vector<std::string> vals;
	str2VecStr_(value_, delim_, vals);

	std::vector<varType> newVals;
	for(auto it=vals.begin(); it!=vals.end(); it++)
		newVals.push_back(convertVariable<varType>(*it));

	if ( logText_ )
	{
		std::ostringstream tempStr;
		tempStr << "l1t::Setting::getVector\tReturning vector with values " << this->getValueAsStr() << " from parameter with id: " << this->getId() << std::endl;
		logText_->append(tempStr.str());
	}
	return newVals;
}

template <class varType> varType Setting::getValue()
{
	
	if ( type_.find("vector") != std::string::npos )
		throw std::runtime_error("The registered type: " + type_ + " is vector so you need to call the getVector method");
	
	if ( logText_ )
	{
		std::ostringstream tempStr;
		tempStr << "l1t::Setting::getValue\tReturning value " << this->getValueAsStr() << " from parameter with id: " << this->getId() << std::endl;
		logText_->append(tempStr.str());
	}
	return convertVariable<varType>(value_);
}

template <class varType> varType TableRow::getRowValue(const std::string& col)
{
	std::map<std::string,int>::const_iterator it = colDict_->find(col);
    if( it == colDict_->end() )
		throw std::runtime_error ("Column " + col + "not found in table " + tableId_);

	if ( logText_ )
	{
		std::ostringstream tempStr;
		tempStr << "l1t::Setting::getRowValue\tReturning value " << convertVariable<varType>(row_->at(it->second)) <<  " from table " << tableId_ << " and row " << this->getRowAsStr() << std::endl;
		logText_->append(tempStr.str());
	}
	return convertVariable<varType>(row_->at(it->second));
}

}
#endif


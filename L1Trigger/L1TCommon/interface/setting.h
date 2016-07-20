#ifndef __setting_h__
#define __setting_h__

#include <vector>
#include <string>
#include <memory>

#include "CondFormats/L1TObjects/interface/LUT.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "L1Trigger/L1TCommon/interface/tools.h"

//boost libraries
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/classic.hpp>
#include <boost/spirit/include/classic_push_back_actor.hpp>

namespace l1t{

class tableRow
{
	public:
		tableRow() {};
		tableRow(const std::vector<std::string>& row, std::string *lt);
		void setTableId(const std::string& id) { tableId_ = id; };
		void setRowTypes(const std::vector<std::string>& types);
		void setRowColumns(const std::vector<std::string>& columns);
		void setLogStringVar(std::string* strVar) {logText_ = strVar;};
		~tableRow() {};
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


class setting
{
	public:
		setting() {};
		setting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole, std::string *lt, const std::string& delim = "");
		setting(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, std::string *lt, const std::string& delim);
		void setProcRole(const std::string& procRole) { procRole_ = procRole; };
		void setValue(const std::string& value);
		void setId(const std::string& id) { id_ = id; } ;
		void setLogStringVar(std::string* strVar) {logText_ = strVar;};
		void addTableRow(const std::string& row, const std::vector<std::string>& types, const std::vector<std::string>& columns);
		void resetTableRows() { tableRows_.clear();};
		std::string getProcRole() { return procRole_; };
		std::string getValueAsStr() { return value_; };
		std::string getType() { return type_; };
		std::string getId() { return id_; } ;
		template <class varType> varType getValue();
		template <class varType> std::vector<varType> getVector();
		std::vector<tableRow>  getTableRows() { return tableRows_; };
		l1t::LUT getLUT(size_t addrWidth = 0, size_t dataWidth = 31, int padding = -1, std::string delim = ","); // if the addrWidth parameter is 0 calculate the address width from the LUT length. 31 is the maximal supported number of bits for the output width of l1t::LUT
		~setting();

		setting& operator=(const setting& aSet);
	private:
		std::string type_, id_, value_, procRole_, delim_;
		std::vector<tableRow> tableRows_;
		std::string* logText_;		
};


template <typename varType> std::vector<varType> setting::getVector()
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
		tempStr << "l1t::setting::getVector\tReturning vector with values " << this->getValueAsStr() << " from parameter with id: " << this->getId() << std::endl;
		logText_->append(tempStr.str());
	}
	return newVals;
}

template <class varType> varType setting::getValue()
{
	
	if ( type_.find("vector") != std::string::npos )
		throw std::runtime_error("The registered type: " + type_ + " is vector so you need to call the getVector method");
	
	if ( logText_ )
	{
		std::ostringstream tempStr;
		tempStr << "l1t::setting::getValue\tReturning value " << this->getValueAsStr() << " from parameter with id: " << this->getId() << std::endl;
		logText_->append(tempStr.str());
	}
	return convertVariable<varType>(value_);
}

template <class varType> varType tableRow::getRowValue(const std::string& col)
{
	std::map<std::string,int>::const_iterator it = colDict_->find(col);
    if( it == colDict_->end() )
		throw std::runtime_error ("Column " + col + "not found in table " + tableId_);

	if ( logText_ )
	{
		std::ostringstream tempStr;
		tempStr << "l1t::setting::getRowValue\tReturning value " << convertVariable<varType>(row_->at(it->second)) <<  " from table " << tableId_ << " and row " << this->getRowAsStr() << std::endl;
		logText_->append(tempStr.str());
	}
	return convertVariable<varType>(row_->at(it->second));
}

}
#endif


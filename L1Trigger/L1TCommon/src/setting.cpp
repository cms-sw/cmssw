#include <strstream>

#include "L1Trigger/L1TCommon/interface/setting.h"

namespace l1t{
	
setting::setting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole, std::string *lt, const std::string& delim) :
type_(type),
id_(id),
value_(value),
procRole_(procRole),
delim_(delim)
{
	if ( delim.empty() )
		delim_ = ",";

	logText_ = lt;

	setValue(value);
}

setting::setting(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, std::string *lt, const std::string& delim) :
type_("table"),
id_(id),
procRole_(procRole),
delim_(delim)
{
    if ( delim.empty() )
    	delim_ = ",";

    logText_ = lt;
	// str2VecStr_(columns, delim_, tableColumns_);

	// str2VecStr_(types, delim_, tableTypes_);

	for (auto it=rows.begin(); it!=rows.end(); ++it)
	{
		// std::vector<std::string> aRow;
		// str2VecStr_(*it, delim_, aRow);

		//tableRow temp(aRow);
		//temp.setTableId(id);
		//temp.setRowTypes(tableTypes_);
		//temp.setRowColumns(tableColumns_);
		tableRows_.push_back(tableRow(str2VecStr_(*it, delim_),logText_));
		tableRows_.back().setTableId(id);
		// tableRows_.back().setRowTypes(tableTypes_);
		// tableRows_.back().setRowColumns(tableColumns_);
		tableRows_.back().setRowTypes(str2VecStr_(types, delim_));
		tableRows_.back().setRowColumns(str2VecStr_(columns, delim_));

	}
}

setting::~setting()
{
	;
}

void setting::setValue(const std::string& value)
{
	if ( type_.find("bool") != std::string::npos )
	{
		std::ostringstream convString;

		if ( type_.find("vector") != std::string::npos )
		{
            if (delim_.empty())
				delim_ = ",";
			
			std::vector<std::string> vals;
			str2VecStr_(value_,delim_, vals);
			
			for(std::vector<std::string>::iterator it=vals.begin(); it!=vals.end(); ++it)
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


l1t::LUT setting::getLUT(size_t addrWidth, size_t dataWidth, int padding, std::string delim)
{
	if ( type_.find("vector:uint") == std::string::npos )
		throw std::runtime_error("Cannot build LUT from type: " + type_ + ". Only vector:uint is allowed.");

	if ( delim.empty() )
		delim = ",";

	std::vector<unsigned int> vec = getVector<unsigned int>();

        // if the addrWidth parameter is 0 calculate the address width from the LUT length
        if (addrWidth == 0) {
		size_t nEntries = vec.size();
		while (nEntries >>= 1) {
			++addrWidth;
		}
        }

	// write the stream to fill the LUT
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

void setting::addTableRow(const std::string& row, const std::vector<std::string>& types, const std::vector<std::string>& columns)
{
	if (type_.find("table") == std::string::npos)
		throw std::runtime_error("You cannot add a table row in type: " + type_ + ". Type is not table.");

	std::vector<std::string> vals;
	str2VecStr_(row, delim_, vals);

	// tableRow tempRow(vals);
	// tempRow.setRowTypes(tableTypes_);
	// tempRow.setRowColumns(tableColumns_);
	tableRows_.push_back(tableRow(vals, logText_));
	tableRows_.back().setRowTypes(types);
	tableRows_.back().setRowColumns(columns);

}

// void setting::setTableTypes(const std::string& types)
// {	
// 	if (type_.find("table") == std::string::npos)
// 		throw std::runtime_error("You cannot set table types in type: " + type_ + ". Type is not table.");
	 

// 	str2VecStr_(types, delim_, tableTypes_);
// }

// void setting::setTableColumns(const std::string& cols)
// {
// 	if (type_.find("table") == std::string::npos)
// 		throw std::runtime_error("You cannot set table columns in type: " + type_ + ". Type is not table.");
	

// 	str2VecStr_(cols, delim_, tableColumns_);
	
// }

tableRow::tableRow(const std::vector<std::string>& row, std::string *lt) 
{ 
	row_ = std::shared_ptr< std::vector<std::string> >(new std::vector<std::string>(row));
	logText_ = lt;
}

void tableRow::setRowColumns(const std::vector<std::string>& columns)
{
    if( columns_.get() == 0 )
    	columns_ = std::shared_ptr< std::vector<std::string> >(new std::vector<std::string>(columns));
    else
        *columns_ = columns;

    if( colDict_.get() == 0 )
        colDict_ = std::shared_ptr< std::map<std::string,int> >(new std::map<std::string,int>());

	colDict_->clear();

	for(unsigned int i=0; i<columns.size(); i++) 
		(*colDict_)[ columns[i] ] = i;
}

void tableRow::setRowTypes(const std::vector<std::string>& types)
{
	if( types_.get() == 0 )
	  types_ = std::shared_ptr< std::vector<std::string> >(new std::vector<std::string>(types));
	else
	  *types_ = types;
}

std::string tableRow::getRowAsStr()
{
	std::ostringstream str;
	for (auto it=row_->begin(); it!=row_->end(); ++it)
		str << *it << " ";

	return str.str();
}

}


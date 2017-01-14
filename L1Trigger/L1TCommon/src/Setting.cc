#include <strstream>

#include "L1Trigger/L1TCommon/interface/Setting.h"

namespace l1t{
	
Setting::Setting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole, std::string *lt, const std::string& delim) :
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

Setting::Setting(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, std::string *lt, const std::string& delim) :
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

		//TableRow temp(aRow);
		//temp.setTableId(id);
		//temp.setRowTypes(tableTypes_);
		//temp.setRowColumns(tableColumns_);
		TableRows_.push_back(TableRow(str2VecStr_(*it, delim_),logText_));
		TableRows_.back().setTableId(id);
		// TableRows_.back().setRowTypes(tableTypes_);
		// TableRows_.back().setRowColumns(tableColumns_);
		TableRows_.back().setRowTypes(str2VecStr_(types, delim_));
		TableRows_.back().setRowColumns(str2VecStr_(columns, delim_));

	}
}

Setting::~Setting()
{
	;
}

void Setting::setValue(const std::string& value)
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


l1t::LUT Setting::getLUT(size_t addrWidth, size_t dataWidth, int padding, std::string delim)
{
	if ( type_.find("vector:uint") == std::string::npos )
		throw std::runtime_error("Cannot build LUT from type: " + type_ + ". Only vector:unsigned int is allowed.");

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

Setting& Setting::operator=(const Setting& aSet)
{
	value_ = aSet.value_;
	id_ = aSet.id_;
	type_ = aSet.type_;
	procRole_ = aSet.procRole_;
	return *this;
}

void Setting::addTableRow(const std::string& row, const std::vector<std::string>& types, const std::vector<std::string>& columns)
{
	if (type_.find("table") == std::string::npos)
		throw std::runtime_error("You cannot add a table row in type: " + type_ + ". Type is not table.");

	std::vector<std::string> vals;
	str2VecStr_(row, delim_, vals);

	// TableRow tempRow(vals);
	// tempRow.setRowTypes(tableTypes_);
	// tempRow.setRowColumns(tableColumns_);
	TableRows_.push_back(TableRow(vals, logText_));
	TableRows_.back().setRowTypes(types);
	TableRows_.back().setRowColumns(columns);

}

// void Setting::setTableTypes(const std::string& types)
// {	
// 	if (type_.find("table") == std::string::npos)
// 		throw std::runtime_error("You cannot set table types in type: " + type_ + ". Type is not table.");
	 

// 	str2VecStr_(types, delim_, tableTypes_);
// }

// void Setting::setTableColumns(const std::string& cols)
// {
// 	if (type_.find("table") == std::string::npos)
// 		throw std::runtime_error("You cannot set table columns in type: " + type_ + ". Type is not table.");
	

// 	str2VecStr_(cols, delim_, tableColumns_);
	
// }

TableRow::TableRow(const std::vector<std::string>& row, std::string *lt) 
{ 
	row_ = std::shared_ptr< std::vector<std::string> >(new std::vector<std::string>(row));
	logText_ = lt;
}

void TableRow::setRowColumns(const std::vector<std::string>& columns)
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

void TableRow::setRowTypes(const std::vector<std::string>& types)
{
	if( types_.get() == 0 )
	  types_ = std::shared_ptr< std::vector<std::string> >(new std::vector<std::string>(types));
	else
	  *types_ = types;
}

std::string TableRow::getRowAsStr()
{
	std::ostringstream str;
	for (auto it=row_->begin(); it!=row_->end(); ++it)
		str << *it << " ";

	return str.str();
}

}


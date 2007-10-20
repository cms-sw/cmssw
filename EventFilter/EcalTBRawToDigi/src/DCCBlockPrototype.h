// Date   : 25/05/2005
// Author : N.Almeida (LIP)

#ifndef DCCTBBLOCKPROTOTYPE_HH
#define DCCTBBLOCKPROTOTYPE_HH


#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <set>
#include <iostream>
#include <iomanip>


class DCCTBDataParser;
class DCCTBDataField;
class DCCTBDataFieldComparator;


class DCCTBBlockPrototype{
	
	public :
		
		DCCTBBlockPrototype(
			DCCTBDataParser * parser, 
			std::string name, 
			ulong * buffer,
			ulong numbBytes, 
			ulong wordsToEndOfEvent, 
			ulong wordEventOffset = 0 
		);

		virtual ~ DCCTBBlockPrototype(){}

		virtual void   parseData();		
		virtual void   increment(ulong numb, std::string msg="");
		virtual void   seeIfIsPossibleToIncrement(ulong numb, std::string msg="");		
		virtual ulong  getDataWord(ulong wordPosition, ulong bitPosition, ulong mask);
		virtual ulong  getDataField(std::string name);
		virtual void   setDataField(std::string name, ulong data);
		
		virtual std::pair<bool,std::string> checkDataField(std::string name, ulong data);
		virtual void displayData(std::ostream & os=std::cout);
		virtual std::pair<bool,std::string> compare(DCCTBBlockPrototype * block);
	
		std::map<std::string,ulong> & errorCounters(){ return errors_; }
		
		// Block Name
		std::string name(){ return name_;}

		// Block Size in Bytes
		ulong size(){ return blockSize_;  }
		
		std::string & errorString(){ return errorString_;}	
		
		//Word Block Offest inside event
		ulong wOffset(){ return wordEventOffset_;}
	
		bool blockError(){return blockError_;}

                /**
                 * Returns data parser
                 */
                DCCTBDataParser *getParser() { return parser_; }
		
	protected :
		
		std::string formatString(std::string myString,ulong minPositions);
		
		ulong * dataP_;
		ulong * beginOfBuffer_;
		
		ulong blockSize_;
		ulong wordCounter_;
		ulong wordEventOffset_;
		ulong wordsToEndOfEvent_;
	
		bool blockError_;
		
		std::string name_;
		std::string errorString_;
		std::string blockString_;
		std::string processingString_;
		
		DCCTBDataParser * parser_;
		
		std::map<std::string,ulong> dataFields_;
		std::map<std::string,ulong> errors_;
		
		std::set<DCCTBDataField *,DCCTBDataFieldComparator> * mapperFields_;
		
	
};



#endif




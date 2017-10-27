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
#include <cstdint>

class DCCTBDataParser;
class DCCTBDataField;
class DCCTBDataFieldComparator;


class DCCTBBlockPrototype{
	
	public :
		
		DCCTBBlockPrototype(
			DCCTBDataParser * parser, 
			std::string name, 
			uint32_t* buffer,
			uint32_t numbBytes, 
			uint32_t wordsToEndOfEvent, 
			uint32_t wordEventOffset = 0 
		);

		virtual ~ DCCTBBlockPrototype(){}

		virtual void   parseData();		
		virtual void   increment(uint32_t numb, std::string msg="");
		virtual void   seeIfIsPossibleToIncrement(uint32_t numb, std::string msg="");		
		virtual uint32_t  getDataWord(uint32_t wordPosition, uint32_t bitPosition, uint32_t mask);
		virtual uint32_t  getDataField(std::string name);
		virtual void   setDataField(std::string name, uint32_t data);
		
		virtual std::pair<bool,std::string> checkDataField(std::string name, uint32_t data);
		virtual void displayData(std::ostream & os=std::cout);
		virtual std::pair<bool,std::string> compare(DCCTBBlockPrototype * block);
	
		std::map<std::string,uint32_t> & errorCounters(){ return errors_; }
		
		// Block Name
		std::string name(){ return name_;}

		// Block Size in Bytes
		uint32_t size(){ return blockSize_;  }
		
		std::string & errorString(){ return errorString_;}	
		
		//Word Block Offest inside event
		uint32_t wOffset(){ return wordEventOffset_;}
	
		bool blockError(){return blockError_;}

                /**
                 * Returns data parser
                 */
                DCCTBDataParser *getParser() { return parser_; }
		
	protected :
		
		std::string formatString(std::string myString,uint32_t minPositions);
		
		uint32_t * dataP_;
		uint32_t * beginOfBuffer_;
		
		uint32_t blockSize_;
		uint32_t wordCounter_;
		uint32_t wordEventOffset_;
		uint32_t wordsToEndOfEvent_;
	
		bool blockError_;
		
		std::string name_;
		std::string errorString_;
		std::string blockString_;
		std::string processingString_;
		
		DCCTBDataParser * parser_;
		
		std::map<std::string,uint32_t> dataFields_;
		std::map<std::string,uint32_t> errors_;
		
		std::set<DCCTBDataField *,DCCTBDataFieldComparator> * mapperFields_;
		
	
};



#endif




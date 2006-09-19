// Date   : 25/05/2005
// Author : N.Almeida (LIP)

#ifndef DCCBLOCKPROTOTYPE_HH
#define DCCBLOCKPROTOTYPE_HH


#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <set>
#include <iostream>
#include <iomanip>

using namespace std;

class DCCDataParser;
class DCCDataField;
class DCCDataFieldComparator;


class DCCBlockPrototype{
	
	public :
		
		DCCBlockPrototype(
			DCCDataParser * parser, 
			string name, 
			ulong * buffer,
			ulong numbBytes, 
			ulong wordsToEndOfEvent, 
			ulong wordEventOffset = 0 
		);

		virtual ~ DCCBlockPrototype(){}

		virtual void   parseData();		
		virtual void   increment(ulong numb, string msg="");
		virtual void   seeIfIsPossibleToIncrement(ulong numb, string msg="");		
		virtual ulong  getDataWord(ulong wordPosition, ulong bitPosition, ulong mask);
		virtual ulong  getDataField(string name);
		virtual void   setDataField(string name, ulong data);
		
		virtual pair<bool,string> checkDataField(string name, ulong data);
		virtual void displayData(ostream & os=cout);
		virtual pair<bool,string> compare(DCCBlockPrototype * block);
	
		map<string,ulong> & errorCounters(){ return errors_; }
		
		// Block Name
		string name(){ return name_;}

		// Block Size in Bytes
		ulong size(){ return blockSize_;  }
		
		string & errorString(){ return errorString_;}	
		
		//Word Block Offest inside event
		ulong wOffset(){ return wordEventOffset_;}
	
		bool blockError(){return blockError_;}

                /**
                 * Returns data parser
                 */
                DCCDataParser *getParser() { return parser_; }
		
	protected :
		
		string formatString(string myString,ulong minPositions);
		
		ulong * dataP_;
		ulong * beginOfBuffer_;
		
		ulong blockSize_;
		ulong wordCounter_;
		ulong wordEventOffset_;
		ulong wordsToEndOfEvent_;
	
		bool blockError_;
		
		string name_;
		string errorString_;
		string blockString_;
		string processingString_;
		
		DCCDataParser * parser_;
		
		map<string,ulong> dataFields_;
		map<string,ulong> errors_;
		
		set<DCCDataField *,DCCDataFieldComparator> * mapperFields_;
		
	
};



#endif




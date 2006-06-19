

#include "EventFilter/EcalRawToDigi/src/DCCBlockPrototype.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataParser.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataMapper.h"
#include "ECALParserBlockException.h"

#include <stdio.h>
#include <iomanip>
#include <sstream>


DCCBlockPrototype::DCCBlockPrototype(DCCDataParser * parser, string name, ulong * buffer, ulong numbBytes, ulong wordsToEndOfEvent,  ulong wordEventOffset ){
	myTab_=14;		
	blockError_        = false;
	parser_            = parser;
	name_              = name;
	dataP_             = buffer ;
	beginOfBuffer_     = buffer ;
	blockSize_         = numbBytes ;
	wordEventOffset_   = wordEventOffset;
	wordsToEndOfEvent_ = wordsToEndOfEvent;
	
	wordCounter_ = 0;
	
	//cout<<endl;
	//cout<<" DEBUG::DCCBlockPrototype:: Block Name                  :   "<<name_<<endl;
	//cout<<" DEBUG::DCCBlockPrototype:: Block size  [bytes]         :   "<<dec<<blockSize_<<endl;
	//cout<<" DEBUG::DCCBlockPrototype:: Number Of Words             :   "<<dec<<blockSize_/4<<endl;
	//cout<<" DEBUG::DCCBlockPrototype:: word event offset           :   "<<dec<<wordEventOffset_<<endl;
	//cout<<" DEBUG::DCCBlockPrototype:: words to end of event       :   "<<dec<<wordsToEndOfEvent_<<endl;
	//cout<<" DEBUG::DCCBlockPrototype:: First Word (*dataP_)        : 0x"<<hex<<(*dataP_)<<endl;
	//cout<<endl;
}


void DCCBlockPrototype::parseData(){
	
	set<DCCDataField *,DCCDataFieldComparator>::iterator it;
	
	//cout<<" DEBUG::DCCBlockPrototype:: fields : "<<dec<<(mapperFields_->size())<<endl;	

	for(it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
		
		//cout<<"DEBUG::parse.. field name : "<<(*it)->name()<<endl;
		
		try{
			ulong data = getDataWord( (*it)->wordPosition() , (*it)->bitPosition(),(*it)->mask());
			dataFields_[(*it)->name()]= data;
			
		}catch( ECALParserBlockException & e){
			
			string localString;
			
			localString +="\n ======================================================================\n"; 		
			localString += string(" ") + name_ + string(" :: out of scope Error :: Unable to get data field : ") + (*it)->name();
			localString += "\n Word position inside block   : " + parser_->getDecString( (*it)->wordPosition() );
			localString += "\n Word position inside event   : " + parser_->getDecString( (*it)->wordPosition() + wordEventOffset_);
			localString += "\n Block Size [bytes]           : " + parser_->getDecString(blockSize_);
			localString += "\n Action -> Stop parsing this block !";
			localString += "\n ======================================================================";
			
			string error("\n Last decoded fields until error : ");
			
			ostringstream a;
		
			try{ displayData(a);}
			catch(ECALParserBlockException &e){}
			string outputErrorString(a.str());
			error += outputErrorString;
			
			errorString_ +=  localString + error;
			
			
			blockError_ = true;
						
			throw( ECALParserBlockException(errorString_) );
			
		}		
	}
	
	//debugg
	//displayData(cout);
	
	
}



ulong DCCBlockPrototype::getDataWord(ulong wordPosition, ulong bitPosition, ulong mask){
	
	if( wordPosition > wordCounter_ ){ increment(wordPosition - wordCounter_);	}
	return ((*dataP_)>>bitPosition)&mask;
	
}



void DCCBlockPrototype::increment(ulong numb,string msg){
	
	seeIfIsPossibleToIncrement(numb,msg);
	dataP_ += numb; wordCounter_ += numb;
}



void DCCBlockPrototype::seeIfIsPossibleToIncrement(ulong numb, string msg){
	if (( ((wordCounter_+numb +1) > blockSize_/4)) ||( wordCounter_ + numb > wordsToEndOfEvent_ )){ 
		string error=string("\n Unable to get next block position (parser stoped!)") +msg;
		error += "\n Decoded fields untill error : ";
		//ostream dataUntilError ;
		ostringstream a;
		string outputErrorString;
		
		
		try{ displayData(a);}
		catch(ECALParserBlockException &e){}
		outputErrorString = a.str();
		error += outputErrorString;
		
		throw ECALParserBlockException(error); 
		blockError_=true;
	}
	
}



void DCCBlockPrototype::displayData(  ostream & os  ){


	set<DCCDataField *,DCCDataFieldComparator>::iterator it;
	//map<string,ulong>::iterator mapIt;
	bool process(true);
	os<<"\n ======================================================================\n"; 
	os<<" Block name : "<<name_<<", size : "<<dec<<blockSize_<<" bytes, event WOffset : "<<wordEventOffset_;
	long currentPosition(0), position(-1);
	
	string dataFieldName;
	for(it = mapperFields_->begin(); it!= mapperFields_->end() && process; it++){
		try{
			dataFieldName =  (*it)->name();
			currentPosition      =  (*it)->wordPosition();
			if( currentPosition != position ){
				os << "\n W["<<setw(5)<<setfill('0')<<currentPosition<<"]" ;
				position = currentPosition; 
			}
			os<<" "<<formatString(dataFieldName,myTab_)<<" = "<<dec<<setw(5)<<getDataField(dataFieldName); 			
		} catch (ECALParserBlockException & e){ process = false; os<<" not able to get data field..."<<dataFieldName<<endl;}
	}
	os<<"\n ======================================================================\n"; 

}





pair<bool,string> DCCBlockPrototype::checkDataField(string name, ulong data){

	string output("");
	pair<bool,string> res;
	bool errorFound(false);
	ulong parsedData =  getDataField(name);
	if( parsedData != data){
		output += string("\n Field : ")+name+(" has value ")+parser_->getDecString( parsedData )+ string(", while ")+parser_->getDecString(data)+string(" is expected"); 	
		
		//debug//////////
		//cout<<output<<endl;
		//////////////////
		
		blockError_ = true;
		errorFound  = true;
	}
	res.first  = !errorFound;
	res.second = output; 
	return res;
}



ulong DCCBlockPrototype::getDataField(string name){
	
	map<string,ulong>::iterator it = dataFields_.find(name);
	if(it == dataFields_.end()){		
		throw ECALParserBlockException( string("\n field named : ")+name+string(" was not found in block ")+name_ );
		blockError_=true;
	}

	return (*it).second;

}



string DCCBlockPrototype::formatString(string myString,ulong minPositions){
	string ret(myString);
	ulong stringSize = ret.size();
	if( minPositions > stringSize ){
		for(ulong i=0;i< minPositions-stringSize;i++){ ret+=" ";}
	}
	return  ret;
}


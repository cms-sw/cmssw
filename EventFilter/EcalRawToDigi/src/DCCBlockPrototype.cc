

#include "DCCBlockPrototype.h"
#include "DCCDataParser.h"
#include "DCCDataMapper.h"
#include "ECALParserBlockException.h"

#include <stdio.h>
#include <iomanip>
#include <sstream>

using namespace std;

DCCBlockPrototype::DCCBlockPrototype(DCCDataParser * parser, string name, ulong * buffer, ulong numbBytes, ulong wordsToEndOfEvent,  ulong wordEventOffset ){
			
	blockError_        = false;
	parser_            = parser;
	name_              = name;
	dataP_             = buffer ;
	beginOfBuffer_     = buffer ;
	blockSize_         = numbBytes ;
	wordEventOffset_   = wordEventOffset;
	wordsToEndOfEvent_ = wordsToEndOfEvent;
	
	wordCounter_ = 0;
	
	/*
	cout<<endl;
	cout<<" DEBUG::DCCBlockPrototype:: Block Name                  :   "<<name_<<endl;
	cout<<" DEBUG::DCCBlockPrototype:: Block size  [bytes]         :   "<<dec<<blockSize_<<endl;
	cout<<" DEBUG::DCCBlockPrototype:: Number Of Words             :   "<<dec<<blockSize_/4<<endl;
	cout<<" DEBUG::DCCBlockPrototype:: word event offset           :   "<<dec<<wordEventOffset_<<endl;
	cout<<" DEBUG::DCCBlockPrototype:: words to end of event       :   "<<dec<<wordsToEndOfEvent_<<endl;
	cout<<" DEBUG::DCCBlockPrototype:: First Word (*dataP_)        : 0x"<<hex<<(*dataP_)<<endl;
	cout<<endl;
	*/
}


void DCCBlockPrototype::parseData(){
  set<DCCDataField *,DCCDataFieldComparator>::iterator it;          //iterator for data fields
	
  //for debug purposes
  //cout << "Starting to parse data in block named : " << endl;
  //cout << " Fields: " << dec << (mapperFields_->size()) << endl;	
  //cout << "\n begin of buffer : "<<hex<<(*beginOfBuffer_)<<endl;
  
  //cycle through mapper fields
  for(it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
	
	/*	
    //for debug purposes
    cout << "\n Field name        : " << (*it)->name();
    cout << "\n Word position     : " <<dec<< (*it)->wordPosition();
    cout << "\n Bit position      : " << (*it)->bitPosition();
    cout << "\n Size              : " << hex << (*it)->mask() << endl;
    cout << "\n data pointer      : " <<hex<<(*dataP_)<<endl;
    cout << "\n wordsToEndOfEvent : " <<dec<<wordsToEndOfEvent_<<endl;
	*/	
		
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
	
	/*
	cout<<"\n DEBUG::DCCBlockPrototype getDataWord method "
	    <<"\n DEBUG::DCCBlockPrototype wordPosition       = "<<wordPosition
	    <<"\n DEBUG::DCCBlockPrototype wordCounter        = "<<wordCounter_
	    <<"\n DEBUG::DCCBlockPrototype going to increment = "<<(wordPosition-wordCounter_)<<endl;
	*/
	if( wordPosition > wordCounter_ ){ increment(wordPosition - wordCounter_);	}

	return ((*dataP_)>>bitPosition)&mask;
	
}



void DCCBlockPrototype::increment(ulong numb,string msg){
	
	seeIfIsPossibleToIncrement(numb,msg);
	dataP_ += numb; wordCounter_ += numb;
}



void DCCBlockPrototype::seeIfIsPossibleToIncrement(ulong numb, string msg){
	
	/*
	cout<<"\n See if is possible to increment numb ="<<dec<<numb<<" msg "<<msg<<endl;
	cout<<" wordCounter_       "<<wordCounter_<<endl;
	cout<<" blockSize          "<<blockSize_<<endl;
	cout<<" wordsToEndOfEvent_ "<<wordsToEndOfEvent_<<endl;
	*/
	
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

	bool process(true);
	os << "\n ======================================================================\n"; 
	os << " Block name : "<<name_<<", size : "<<dec<<blockSize_<<" bytes, event WOffset : "<<wordEventOffset_;
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
			os<<" "<<formatString(dataFieldName,14)<<" = "<<dec<<setw(5)<<getDataField(dataFieldName); 			
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






void DCCBlockPrototype::setDataField(string name, ulong data){
 set<DCCDataField *,DCCDataFieldComparator>::iterator it;          //iterator for data fields
  bool fieldFound(false);
  for(it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
  	if( ! ((*it)->name()).compare(name) ){ fieldFound = true; }
  }
  
  if(fieldFound){ dataFields_[name]= data;}
  else{ 
  	throw  ECALParserBlockException( string("\n field named : ")+name+string(" was not found in block ")+name_ );
  }
  
}






pair<bool,string> DCCBlockPrototype::compare(DCCBlockPrototype * block){
	
	
	pair<bool,string> ret(true,"");
	
	
	set<DCCDataField *,DCCDataFieldComparator>::iterator it;
	stringstream out;
	

	
	out<<"\n ======================================================================"; 
    out<<"\n ORIGINAL BLOCK    : ";
	out<<"\n Block name : "<<name_<<", size : "<<dec<<blockSize_<<" bytes, event WOffset : "<<wordEventOffset_;
	out<<"\n COMPARISION BLOCK : ";
	out<<"\n Block name : "<<(block->name())<<", size : "<<dec<<(block->size())<<" bytes, event WOffset : "<<(block->wOffset());
	out<<"\n =====================================================================";
	
	
	if( block->name() != name_ ){
		ret.first  = false;
		out<<"\n ERROR >> It is not possible to compare blocks with different names ! ";
		ret.second += out.str();
		return ret;
	}
	
	if( block->size() != blockSize_ ){
		ret.first  = false;
		out<<"\n WARNING >> Blocks have different sizes "
		   <<"\n WARNING >> Comparision will be carried on untill possible";	
	}
	

	if( block->wOffset()!= wordEventOffset_){
		ret.first  = false;
		out<<"\n WARNING >> Blocks have different word offset within the event ";	
	}

		
	string dataFieldName;
	
	for(it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
		
		dataFieldName    =  (*it)->name();
		
		ulong aValue, bValue;
			
		//Access original block data fields /////////////////////////////////////////////////////
		try{ aValue = getDataField(dataFieldName); }
		
		catch(ECALParserBlockException &e ){
			ret.first   = false;
			out<<"\n ERROR ON ORIGINAL BLOCK unable to get data field :"<<dataFieldName;
			out<<"\n Comparision was stoped ! ";
			ret.second += out.str();
			return ret;
		}
		/////////////////////////////////////////////////////////////////////////////////////////
			
		//Access comparision block data fields ///////////////////////////////////////////////////////
		try{ bValue = block->getDataField(dataFieldName); }
		catch(ECALParserBlockException &e ){
			ret.first  = false;
			out<<"\n ERROR ON COMPARISION BLOCK unable to get data field :"<<dataFieldName
			   <<"\n Comparision was stoped ! ";
			ret.second += out.str();
			return ret;
		}
		////////////////////////////////////////////////////////////////////////////////////////////////
		
		
		//cout<<"\n data Field name "<<dataFieldName<<endl;
		//cout<<"\n aValue "<<dec<<aValue<<endl;
		//cout<<"\n bValue "<<dec<<bValue<<endl;
		
		
			
		// Compare values 
		if( aValue != bValue ){
			ret.first = false;
			out<<"\n Data Field : "<<dataFieldName
			   <<"\n ORIGINAL BLOCK value : "<<dec<<setw(5)<<aValue<<" , COMPARISION BLOCK value : "<<dec<<setw(5)<<bValue;
		    //cout<<"\n  debug... "<<out<<endl; 
		}
	}
	out<<"\n ======================================================================\n"; 
	ret.second = out.str();
	
	return ret;
	
}

#include "DCCBlockPrototype.h"
#include "DCCDataParser.h"
#include "DCCDataMapper.h"
#include "ECALParserBlockException.h"

#include <cstdio>
#include <sstream>


DCCTBBlockPrototype::DCCTBBlockPrototype(DCCTBDataParser * parser, std::string name, uint32_t * buffer, uint32_t numbBytes, uint32_t wordsToEndOfEvent,  uint32_t wordEventOffset ){
			
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
	std::cout<<std::endl;
	std::cout<<" DEBUG::DCCTBBlockPrototype:: Block Name                  :   "<<name_<<std::endl;
	std::cout<<" DEBUG::DCCTBBlockPrototype:: Block size  [bytes]         :   "<<std::dec<<blockSize_<<std::endl;
	std::cout<<" DEBUG::DCCTBBlockPrototype:: Number Of Words             :   "<<std::dec<<blockSize_/4<<std::endl;
	std::cout<<" DEBUG::DCCTBBlockPrototype:: word event offset           :   "<<std::dec<<wordEventOffset_<<std::endl;
	std::cout<<" DEBUG::DCCTBBlockPrototype:: words to end of event       :   "<<std::dec<<wordsToEndOfEvent_<<std::endl;
	std::cout<<" DEBUG::DCCTBBlockPrototype:: First Word (*dataP_)        : 0x"<<hex<<(*dataP_)<<std::endl;
	std::cout<<std::endl;
	*/
}


void DCCTBBlockPrototype::parseData(){
  std::set<DCCTBDataField *,DCCTBDataFieldComparator>::iterator it;          //iterator for data fields
	
  //for debug purposes
  //std::cout << "Starting to parse data in block named : " << std::endl;
  //std::cout << " Fields: " << std::dec << (mapperFields_->size()) << std::endl;	
  //std::cout << "\n begin of buffer : "<<hex<<(*beginOfBuffer_)<<std::endl;
  
  //cycle through mapper fields
  for(it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
	
	/*	
    //for debug purposes
    std::cout << "\n Field name        : " << (*it)->name();
    std::cout << "\n Word position     : " <<std::dec<< (*it)->wordPosition();
    std::cout << "\n Bit position      : " << (*it)->bitPosition();
    std::cout << "\n Size              : " << hex << (*it)->mask() << std::endl;
    std::cout << "\n data pointer      : " <<hex<<(*dataP_)<<std::endl;
    std::cout << "\n wordsToEndOfEvent : " <<std::dec<<wordsToEndOfEvent_<<std::endl;
	*/	
		
    try{
      uint32_t data = getDataWord( (*it)->wordPosition() , (*it)->bitPosition(),(*it)->mask());
      dataFields_[(*it)->name()]= data;
      	
    }catch( ECALTBParserBlockException & e){
			
      std::string localString;
      
      localString +="\n ======================================================================\n"; 		
      localString += std::string(" ") + name_ + std::string(" :: out of scope Error :: Unable to get data field : ") + (*it)->name();
      localString += "\n Word position inside block   : " + parser_->getDecString( (*it)->wordPosition() );
      localString += "\n Word position inside event   : " + parser_->getDecString( (*it)->wordPosition() + wordEventOffset_);
      localString += "\n Block Size [bytes]           : " + parser_->getDecString(blockSize_);
      localString += "\n Action -> Stop parsing this block !";
      localString += "\n ======================================================================";
			
      std::string error("\n Last decoded fields until error : ");
			
      std::ostringstream a;
      
      try{ displayData(a);}
      catch(ECALTBParserBlockException &e){}
     
      std::string outputErrorString(a.str());
      error += outputErrorString;
			
      errorString_ +=  localString + error;
       
      blockError_ = true;
      
      throw( ECALTBParserBlockException(errorString_) );
      
    }		
  }
  
  //debugg
  //displayData(std::cout);
}



uint32_t DCCTBBlockPrototype::getDataWord(uint32_t wordPosition, uint32_t bitPosition, uint32_t mask){
	
	/*
	std::cout<<"\n DEBUG::DCCTBBlockPrototype getDataWord method "
	    <<"\n DEBUG::DCCTBBlockPrototype wordPosition       = "<<wordPosition
	    <<"\n DEBUG::DCCTBBlockPrototype wordCounter        = "<<wordCounter_
	    <<"\n DEBUG::DCCTBBlockPrototype going to increment = "<<(wordPosition-wordCounter_)<<std::endl;
	*/
	if( wordPosition > wordCounter_ ){ increment(wordPosition - wordCounter_);	}

	return ((*dataP_)>>bitPosition)&mask;
	
}



void DCCTBBlockPrototype::increment(uint32_t numb,std::string msg){
	
	seeIfIsPossibleToIncrement(numb,msg);
	dataP_ += numb; wordCounter_ += numb;
}



void DCCTBBlockPrototype::seeIfIsPossibleToIncrement(uint32_t numb, std::string msg){
	
	/*
	std::cout<<"\n See if is possible to increment numb ="<<std::dec<<numb<<" msg "<<msg<<std::endl;
	std::cout<<" wordCounter_       "<<wordCounter_<<std::endl;
	std::cout<<" blockSize          "<<blockSize_<<std::endl;
	std::cout<<" wordsToEndOfEvent_ "<<wordsToEndOfEvent_<<std::endl;
	*/
	
	if (( ((wordCounter_+numb +1) > blockSize_/4)) ||( wordCounter_ + numb > wordsToEndOfEvent_ )){ 
			
		std::string error=std::string("\n Unable to get next block position (parser stoped!)") +msg;
		error += "\n Decoded fields untill error : ";
		//ostream dataUntilError ;
		std::ostringstream a;
		std::string outputErrorString;
		
		
		try{ displayData(a);}
		catch(ECALTBParserBlockException &e){}
		outputErrorString = a.str();
		error += outputErrorString;
		
		throw ECALTBParserBlockException(error); 
		blockError_=true;
	}
	
}



void DCCTBBlockPrototype::displayData(  std::ostream & os  ){

  
  std::set<DCCTBDataField *,DCCTBDataFieldComparator>::iterator it;

	bool process(true);
	os << "\n ======================================================================\n"; 
	os << " Block name : "<<name_<<", size : "<<std::dec<<blockSize_<<" bytes, event WOffset : "<<wordEventOffset_;
	long currentPosition(0), position(-1);
	
	std::string dataFieldName;
	for(it = mapperFields_->begin(); it!= mapperFields_->end() && process; it++){
		try{
			dataFieldName =  (*it)->name();
			currentPosition      =  (*it)->wordPosition();
			if( currentPosition != position ){
			  os << "\n W["<<std::setw(5)<<std::setfill('0')<<currentPosition<<"]" ;
				position = currentPosition; 
			}
			os<<" "<<formatString(dataFieldName,14)<<" = "<<std::dec<<std::setw(5)<<getDataField(dataFieldName); 			
		} catch (ECALTBParserBlockException & e){ process = false; os<<" not able to get data field..."<<dataFieldName<<std::endl;}
	}
	os<<"\n ======================================================================\n"; 

}





std::pair<bool,std::string> DCCTBBlockPrototype::checkDataField(std::string name, uint32_t data){

	std::string output("");
	std::pair<bool,std::string> res;
	bool errorFound(false);
	uint32_t parsedData =  getDataField(name);
	if( parsedData != data){
		output += std::string("\n Field : ")+name+(" has value ")+parser_->getDecString( parsedData )+ std::string(", while ")+parser_->getDecString(data)+std::string(" is expected"); 	
		
		//debug//////////
		//std::cout<<output<<std::endl;
		//////////////////
		
		blockError_ = true;
		errorFound  = true;
	}
	res.first  = !errorFound;
	res.second = output; 
	return res;
}



uint32_t DCCTBBlockPrototype::getDataField(std::string name){
	
	std::map<std::string,uint32_t>::iterator it = dataFields_.find(name);
	if(it == dataFields_.end()){		
		throw ECALTBParserBlockException( std::string("\n field named : ")+name+std::string(" was not found in block ")+name_ );
		blockError_=true;
	}

	return (*it).second;

}



std::string DCCTBBlockPrototype::formatString(std::string myString,uint32_t minPositions){
	std::string ret(myString);
	uint32_t stringSize = ret.size();
	if( minPositions > stringSize ){
		for(uint32_t i=0;i< minPositions-stringSize;i++){ ret+=" ";}
	}
	return  ret;

}






void DCCTBBlockPrototype::setDataField(std::string name, uint32_t data){
  std::set<DCCTBDataField *,DCCTBDataFieldComparator>::iterator it;          //iterator for data fields
  bool fieldFound(false);
  for(it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
  	if( ! ((*it)->name()).compare(name) ){ fieldFound = true; }
  }
  
  if(fieldFound){ dataFields_[name]= data;}
  else{ 
  	throw  ECALTBParserBlockException( std::string("\n field named : ")+name+std::string(" was not found in block ")+name_ );
  }
  
}






std::pair<bool,std::string> DCCTBBlockPrototype::compare(DCCTBBlockPrototype * block){
	
	
	std::pair<bool,std::string> ret(true,"");
	
	
	std::set<DCCTBDataField *,DCCTBDataFieldComparator>::iterator it;
	std::stringstream out;
	

	
	out<<"\n ======================================================================"; 
    out<<"\n ORIGINAL BLOCK    : ";
    out<<"\n Block name : "<<name_<<", size : "<<std::dec<<blockSize_<<" bytes, event WOffset : "<<wordEventOffset_;
	out<<"\n COMPARISION BLOCK : ";
	out<<"\n Block name : "<<(block->name())<<", size : "<<std::dec<<(block->size())<<" bytes, event WOffset : "<<(block->wOffset());
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

		
	std::string dataFieldName;
	
	for(it = mapperFields_->begin(); it!= mapperFields_->end(); it++){
		
		dataFieldName    =  (*it)->name();
		
		uint32_t aValue, bValue;
			
		//Access original block data fields /////////////////////////////////////////////////////
		try{ aValue = getDataField(dataFieldName); }
		
		catch(ECALTBParserBlockException &e ){
			ret.first   = false;
			out<<"\n ERROR ON ORIGINAL BLOCK unable to get data field :"<<dataFieldName;
			out<<"\n Comparision was stoped ! ";
			ret.second += out.str();
			return ret;
		}
		/////////////////////////////////////////////////////////////////////////////////////////
			
		//Access comparision block data fields ///////////////////////////////////////////////////////
		try{ bValue = block->getDataField(dataFieldName); }
		catch(ECALTBParserBlockException &e ){
			ret.first  = false;
			out<<"\n ERROR ON COMPARISION BLOCK unable to get data field :"<<dataFieldName
			   <<"\n Comparision was stoped ! ";
			ret.second += out.str();
			return ret;
		}
		////////////////////////////////////////////////////////////////////////////////////////////////
		
		
		//std::cout<<"\n data Field name "<<dataFieldName<<std::endl;
		//std::cout<<"\n aValue "<<std::dec<<aValue<<std::endl;
		//std::cout<<"\n bValue "<<std::dec<<bValue<<std::endl;
		
		
			
		// Compare values 
		if( aValue != bValue ){
			ret.first = false;
			out<<"\n Data Field : "<<dataFieldName
			   <<"\n ORIGINAL BLOCK value : "<<std::dec<<std::setw(5)<<aValue<<" , COMPARISION BLOCK value : "<<std::dec<<std::setw(5)<<bValue;
		    //std::cout<<"\n  debug... "<<out<<std::endl; 
		}
	}
	out<<"\n ======================================================================\n"; 
	ret.second = out.str();
	
	return ret;
	
}

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/Utilities/interface/CSVDataLineParser.h"
#include "CondTools/Utilities/interface/CSVHeaderLineParser.h"
#include "CondTools/Utilities/interface/CSVBlankLineParser.h"
#include "../interface/OptAlignDataConverter.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"

#include <fstream>
#include <iostream>

OptAlignDataConverter::OptAlignDataConverter(const edm::ParameterSet& iConfig):m_inFileName( iConfig.getUntrackedParameter< std::string >("inputFile") ){}

void OptAlignDataConverter::endJob()
{
  std::ifstream myfile(m_inFileName.c_str());
  if ( !myfile.is_open() ) throw cms::Exception("unable to open file");
  OpticalAlignments* myobj=new OpticalAlignments;
  std::vector<std::string> fieldNames,fieldTypes;
  CSVHeaderLineParser headerParser;
  CSVDataLineParser dataParser;
  int counter=0;
  while (! myfile.eof() ){
    std::string line;
    std::getline (myfile,line);
    std::cout << " line read " << line << std::endl;

    CSVBlankLineParser blank;
    if(blank.isBlank(line)){
      continue;
    }
    if(counter<2){//two lines of header
      if(counter==0) {
	if(!headerParser.parse(line)) {
	  throw cms::Exception("unable to parse header: ")<<line;
	}
	fieldNames=headerParser.result();
      }
      if(counter==1) {
	if(!headerParser.parse(line)) {
	}
	fieldTypes=headerParser.result();
	int idx=0;
	for(std::vector<std::string>::iterator it=fieldTypes.begin(); it!=fieldTypes.end(); ++it, ++idx){
	  std::cout<<fieldNames[idx]<<":"<<*it<<std::endl;
	  m_fieldMap.push_back(fieldNames[idx],*it);
	}
      }
      ++counter;
      continue;
    }
    if(!dataParser.parse(line)) {
      throw cms::Exception("unable to parse data :")<<line;
    }
    std::vector<boost::any> result=dataParser.result();
    OpticalAlignInfo* oaInfo = new OpticalAlignInfo;
    std::string theLastExtraEntryName;
    OpticalAlignParam* theLastExtraEntry = 0;
    std::vector<OpticalAlignParam*> theExtraEntries;

    int idx=0;
    for(std::vector<boost::any>::iterator it=result.begin(); it!=result.end(); ++it, ++idx){
      std::string fieldName=m_fieldMap.fieldName(idx);
      //      std::cout << " idx " << idx << " = " << fieldName << std::endl;
      if(fieldName=="ID"){
	//std::cout<<"fieldName "<<fieldName<<" field type "<<m_fieldMap.fieldTypeName(idx)<<std::endl;
	if( m_fieldMap.fieldType(idx)!= typeid(int) ) throw cond::Exception("unexpected type");
	oaInfo->ID_=boost::any_cast<int>(*it);
      } else if(fieldName=="type"){
	//std::cout<<"fieldName "<<fieldName<<" field type "<<m_fieldMap.fieldTypeName(idx)<<std::endl;
	if( m_fieldMap.fieldType(idx)!= typeid(std::string) ) throw cond::Exception("unexpected type");
	oaInfo->type_=boost::any_cast<std::string>(*it);
      }
      if(fieldName=="name"){
	//std::cout<<"fieldName "<<fieldName<<" field type "<<m_fieldMap.fieldTypeName(idx)<<std::endl;
	if( m_fieldMap.fieldType(idx)!= typeid(std::string) ) throw cond::Exception("unexpected type");
	oaInfo->name_=boost::any_cast<std::string>(*it);
      }
      if(fieldName=="centre_X"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->x_.value_=boost::any_cast<double>(*it);
      } else if(fieldName=="centre_Y"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->y_.value_=boost::any_cast<double>(*it);
      } else if(fieldName=="centre_Z"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->z_.value_=boost::any_cast<double>(*it);
      } else if(fieldName=="centre_sigma_X" || fieldName=="centre_error_X"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->x_.error_=boost::any_cast<double>(*it);
      } else if(fieldName=="centre_sigma_Y" || fieldName=="centre_error_Y"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->y_.error_=boost::any_cast<double>(*it);
      } else if(fieldName=="centre_sigma_Z" || fieldName=="centre_error_Z"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->z_.error_=boost::any_cast<double>(*it);
      } else if(fieldName=="angles_X"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->angx_.value_=boost::any_cast<double>(*it);
      } else if(fieldName=="angles_Y"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->angy_.value_=boost::any_cast<double>(*it);
      } else if(fieldName=="angles_Z"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->angz_.value_=boost::any_cast<double>(*it);
      } else if(fieldName=="angles_sigma_X" || fieldName=="angles_error_X"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->angx_.error_=boost::any_cast<double>(*it);
      } else if(fieldName=="angles_sigma_Y" || fieldName=="angles_error_Y"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->angy_.error_=boost::any_cast<double>(*it);
      } else if(fieldName=="angles_sigma_Z" || fieldName=="angles_error_Z"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	oaInfo->angz_.error_=boost::any_cast<double>(*it);
      } else if( fieldName.substr(0,5) == "param" && fieldName.substr(fieldName.length()-4,4) == "name"){
	if( m_fieldMap.fieldType(idx) != typeid(std::string) ) throw cond::Exception("unexpected type");
	std::string name = boost::any_cast<std::string>(*it);
	theLastExtraEntry = new OpticalAlignParam;
	theExtraEntries.push_back( theLastExtraEntry );
	theLastExtraEntry->name_ = name;
	if( name != "None" ){
	  theLastExtraEntryName = name;
	}
      } else if( fieldName.substr(0,5) == "param" && fieldName.substr(fieldName.length()-5,5) == "value"){
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	if( boost::any_cast<double>(*it) != -9.999E9 ){
	  if( theLastExtraEntryName == "None" ) throw cond::Exception("unexpected type: setting a value != -9.999E9 for a parameter that is 'None' ");
	  theLastExtraEntry->value_ = boost::any_cast<double>(*it);
	}
      } else if( fieldName.substr(0,5) == "param" && ( fieldName.substr(fieldName.length()-5,5) == "sigma" || fieldName.substr(fieldName.length()-5,5) == "error") ) {
	if( m_fieldMap.fieldType(idx) != typeid(float) ) throw cond::Exception("unexpected type");
	if( boost::any_cast<double>(*it) != -9.999E9 ){
	  if( theLastExtraEntryName == "None" ) throw cond::Exception("unexpected type: setting an error != -9.999E9 for a parameter that is 'None' ");
	  theLastExtraEntry->error_ = boost::any_cast<double>(*it);
	}
      } // end loop to one oaInfo
    } 

    for( size_t kk = 0; kk < theExtraEntries.size(); kk++) {
      oaInfo->extraEntries_.push_back( *(theExtraEntries[kk]) );
      delete theExtraEntries[kk];
    }

    myobj->opticalAlignments_.push_back(*oaInfo);
    std::cout << " OptAlignInfo read " << *oaInfo << std::endl;

    delete oaInfo;
    ++counter;
  }
  myfile.close();
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( mydbservice.isAvailable() ){
    try{
      mydbservice->newValidityForNewPayload<OpticalAlignments>(myobj,mydbservice->endOfTime());
    }catch(const cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(const std::exception& er){
      std::cout<<"caught std::exception "<<er.what()<<std::endl;
    }catch(...){
      std::cout<<"Funny error"<<std::endl;
    }
  }
  std::cout << "@@@@ OPTICALALIGNMENTS WRITTEN TO DB " << *myobj << std::endl;
}

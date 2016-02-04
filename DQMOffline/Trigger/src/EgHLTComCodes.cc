#include "DQMOffline/Trigger/interface/EgHLTComCodes.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace egHLT;

void ComCodes::setCode(const char* descript,int code)
{
  bool found=false;
  for(size_t i=0;i<_codeDefs.size() && !found;i++){
    if(_codeDefs[i].first.compare(descript)==0) found=true;
  }
  if(!found) _codeDefs.push_back(std::pair<std::string,int>(descript,code));
 
  //_codeDefs[descript] = code;
}



int ComCodes::getCode(const char* descript)const
{ 
  //first copy the character string to a local array so we can manipulate it
  char localDescript[256];
  strcpy(localDescript,descript);
  
  int code = 0x0000; 
  char* codeKey = strtok(localDescript,":");
  //  std::map<std::string,int> ::const_iterator mapIt;
  while(codeKey!=NULL){
    bool found=false;

    for(size_t i=0;i<_codeDefs.size() && !found;i++){
      if(_codeDefs[i].first.compare(codeKey)==0){
 	found=true;
 	code |= _codeDefs[i].second;

       }
    }
   
    if(!found)  edm::LogWarning("EgHLTComCodes") <<"ComCodes::getCode : Error, Key "<<codeKey<<" not found (likely mistyped, practical upshot is the selection is not what you think it is)";//<<std::endl;
    codeKey = strtok(NULL,":"); //getting new substring
    
  }
  return code;
}

bool ComCodes::keyComp(const std::pair<std::string,int>& lhs,const std::pair<std::string,int>& rhs)
{
  return lhs.first < rhs.first;
}

void ComCodes::getCodeName(int code,std::string& id)const
{
  id.clear();
  for(size_t i=0;i<_codeDefs.size();i++){ 
    if((code&_codeDefs[i].second)==_codeDefs[i].second){
      if(!id.empty()) id+=":";//seperating entries by a ':'
      id+=_codeDefs[i].first;
    }
    
  }
 
}

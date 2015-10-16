#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

const int egHLT::TrigCodes::maxNrBits_;

using namespace egHLT;

TrigCodes* TrigCodes::makeCodes(std::vector<std::string>& filterNames)
{
  TrigCodes *p = new TrigCodes();

  for (size_t i=0;i<filterNames.size();i++) { 
    p->setCode(filterNames[i].c_str(),i);
  }
  p->sort();

  return p;
}

void TrigCodes::setCode(const char* descript,int bitNr)
{
  if(bitNr < maxNrBits_){
    TrigBitSet code;
    code.set(bitNr);
    setCode(descript,code);
  }else{
    edm::LogWarning("TrigCodes::TrigBitSetMap") <<" Warning, trying to store at bit "<<bitNr<<" but max nr bits is "<<maxNrBits_;
  }
}

void TrigCodes::setCode(const char* descript,TrigBitSet code)
{
  bool found=false;
  for(size_t i=0;i<codeDefs_.size() && !found;i++){
    if(codeDefs_[i].first.compare(descript)==0) found=true;
  }
  if(!found) codeDefs_.push_back(std::pair<std::string,TrigBitSet>(descript,code));
  //_codeDefs[descript] = code;
}



TrigCodes::TrigBitSet TrigCodes::getCode(const char* descript)const
{ 
  //first copy the character string to a local array so we can manipulate it
  char localDescript[512];
  strcpy(localDescript,descript);
  
  TrigBitSet code; 
  char* codeKey = strtok(localDescript,":");
  //  std::map<std::string,int> ::const_iterator mapIt;
  while(codeKey!=NULL){
    bool found=false;

    for(size_t i=0;i<codeDefs_.size() && !found;i++){
      if(codeDefs_[i].first.compare(codeKey)==0){
 	found=true;
 	code |= codeDefs_[i].second;

       }
    }
   
    //  if(!found)  edm::LogError("TrigCodes::TrigBitSetMap") <<"TrigCodes::TrigBitSetMap::getCode : Error, Key "<<codeKey<<" not found";
    codeKey = strtok(NULL,":"); //getting new substring
    
  }
  return code;
}

bool TrigCodes::keyComp(const std::pair<std::string,TrigBitSet>& lhs,const std::pair<std::string,TrigBitSet>& rhs)
{
  return lhs.first < rhs.first;
}

void TrigCodes::getCodeName(TrigBitSet code,std::string& id)const
{
  id.clear();
  for(size_t i=0;i<codeDefs_.size();i++){ 
    if((code&codeDefs_[i].second)==codeDefs_[i].second){
      if(!id.empty()) id+=":";//seperating entries by a ':'
      id+=codeDefs_[i].first;
    }
    
  }
 
}

void TrigCodes::printCodes()
{
  std::ostringstream msg;
  msg <<" trig bits defined: "<<std::endl;
  for(size_t i=0;i<codeDefs_.size();i++) msg <<" key : "<<codeDefs_[i].first<<" bit "<<codeDefs_[i].second<<std::endl;
  edm::LogInfo("TrigCodes") << msg.str();
 
}

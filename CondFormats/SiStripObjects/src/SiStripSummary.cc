#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"

		
SiStripSummary::SiStripSummary(std::vector<std::string>& userDBContent, std::string tag)
{
  userDBContent_ = userDBContent;
  tag_ = tag;
  runNr_ = 0;
  timeValue_ = 0;
}

SiStripSummary::SiStripSummary(const SiStripSummary& input)
{

  userDBContent_ = input.getUserDBContent();
  runNr_ = input.getTimeValue();
  timeValue_ = input.getRunNr();
  v_sum.clear();
  indexes.clear();
  v_sum.insert(v_sum.end(),input.v_sum.begin(),input.v_sum.end());
  indexes.insert(indexes.end(),input.indexes.begin(),input.indexes.end());
}



bool SiStripSummary::put(const uint32_t& DetId,InputVector& input) {

  Registry::iterator p 	= 	std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripSummary::StrictWeakOrdering());
  if (p!=indexes.end() 	&& 	p->detid==DetId) return false;

  if (input.size() != userDBContent_.size())  throw cms::Exception("")<< "[SiStripSummary::put] : size of db object doesn't match the declared content"; 
	
  DetRegistry detregistry;
  detregistry.detid  = DetId;
  detregistry.ibegin = v_sum.size();
  detregistry.iend   = v_sum.size()+input.size();
  indexes.insert(p,detregistry);
  v_sum.insert(v_sum.begin()+v_sum.size(),input.begin(),input.end());
	
  return true;
}

bool SiStripSummary::put(const uint32_t& DetId, InputVector &input, std::vector<std::string>& userContent ) {

  Registry::iterator p 	= 	std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripSummary::StrictWeakOrdering());
  
  if(p==indexes.end() || p->detid!=DetId){
    //First request for the given DetID
    //Create entries for all the declared userDBContent
    //and fill for the provided userContent
 
    std::cout << "test Summary : first entry for detid " << DetId << std::endl;
    DetRegistry detregistry;
    detregistry.detid  = DetId;
    detregistry.ibegin = v_sum.size();
    detregistry.iend   = v_sum.size()+userDBContent_.size();
    indexes.insert(p,detregistry);
    InputVector tmp(userDBContent_.size(),-9999);

    for(size_t i=0;i<userContent.size();++i)
      tmp[getPosition(userContent[i])]=input[i];
    
    v_sum.insert(v_sum.end(),tmp.begin(),tmp.end());
  } else {

    if (p->detid==DetId){
      // I should already find the entries 
      //fill for the provided userContent

      std::cout << "test Summary : another entry for detid " << DetId << std::endl;

      for(size_t i=0;i<userContent.size();++i)
	v_sum[p->ibegin+getPosition(userContent[i])]=input[i];
    }
  }
	
  return true;
}


bool SiStripSummary::put(TrackerRegion region, InputVector &input) {

  uint32_t fakeDet = region;
  return put(fakeDet, input);
}

bool SiStripSummary::put(TrackerRegion region, InputVector &input, std::vector<std::string>& userContent ) {

  uint32_t fakeDet = region;
  return put(fakeDet, input,userContent);
}



bool SiStripSummary::put(const uint32_t& DetId, float input) {

  Registry::iterator p 	= 	std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripSummary::StrictWeakOrdering());
  if (p!=indexes.end() 	&& 	p->detid==DetId) return false;
	
  DetRegistry detregistry;
  detregistry.detid  = DetId;
  detregistry.ibegin = v_sum.size();
  detregistry.iend   = v_sum.size()+1;
  indexes.insert(p,detregistry);
  v_sum.insert(v_sum.end(),input);
  return true;
}




const SiStripSummary::Range SiStripSummary::getRange(const uint32_t& DetId) const {

  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripSummary::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiStripSummary::Range(v_sum.end(),v_sum.end()); 
  else 
    return SiStripSummary::Range(v_sum.begin()+p->ibegin,v_sum.begin()+p->iend);
}


std::vector<uint32_t> SiStripSummary::getDetIds() const {
  // returns vector of DetIds in map
  std::vector<uint32_t> DetIds_;
  SiStripSummary::RegistryIterator begin = indexes.begin();
  SiStripSummary::RegistryIterator end   = indexes.end();
  for (SiStripSummary::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
  return DetIds_;
}


void SiStripSummary::setData(float summaryInfo, std::vector<float>& v){
  v.push_back(summaryInfo) ;
}


const size_t SiStripSummary::getPosition(std::string elementName) const
{
  std::vector<std::string>::const_iterator it = find(userDBContent_.begin(),userDBContent_.end(),elementName);  
  std::vector<std::string>::difference_type pos = -1;
  if (it != userDBContent_.end()) pos = it - userDBContent_.begin();
  //else  throw cms::Exception("[SiStripSummary::getPosition]")
  //	  << " attempting to retrieve non existing historic DB object : "<< elementName;
  else std::cout << "attempting to retrieve non existing historic DB object : "<< elementName <<std::endl;
  return pos;  
}   



void  SiStripSummary::setObj(const uint32_t& detID, std::string elementName, float value) 
{
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),detID,SiStripSummary::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=detID) 
    {
      throw cms::Exception("")
	<<"not allowed to modify "<< elementName << " in historic DB - SummaryObj needs to be available first !";
    }

  const SiStripSummary::Range range = getRange(detID);
   
  std::vector<float>::const_iterator it = range.first+getPosition(elementName);
  std::vector<float>::difference_type pos = -1;
  if (it != v_sum.end()){ 
    pos = it - v_sum.begin();
    v_sum.at(pos) = value; }  
}


void  SiStripSummary::print() 
{
  std::cout << "Nr. of detector elements in SiStripSummary object is " << indexes.size() 
	    << " RunNr= " << runNr_ 
	    << " timeValue= " << timeValue_ 
	    << std::endl;
}



void SiStripSummary::setSummaryObj(const uint32_t& detID, std::vector<float>& SummaryObj)
{
  put(detID, SummaryObj);
}




float SiStripSummary::getSummaryObj(uint32_t& detID, std::string elementName) const
{
  const SiStripSummary::Range range = getRange(detID);
  if (getPosition(elementName) != -1) return *((range.first)+getPosition(elementName));
   
  else return -999;
}


std::vector<float> SiStripSummary::getSummaryObj(uint32_t& detID, std::vector<std::string> list) const
{  
  std::vector<float> SummaryObj;
  const SiStripSummary::Range range = getRange(detID);
  for (unsigned int i=0; i<list.size(); i++)
    { if (getPosition(list.at(i))!=-1) SummaryObj.push_back(*((range.first)+getPosition(list.at(i))));
    else SummaryObj.push_back(-999.);}
  return SummaryObj;
}



std::vector<float> SiStripSummary::getSummaryObj(uint32_t& detID) const
{  
  std::vector<float> SummaryObj;
  const SiStripSummary::Range range = getRange(detID);
  for (unsigned int i=0; i<userDBContent_.size(); i++) SummaryObj.push_back(*((range.first)+i));
  return SummaryObj;
}



std::vector<float> SiStripSummary::getSummaryObj() const
{
  return v_sum;
}


std::vector<float> SiStripSummary::getSummaryObj(std::string elementName) const
{
  std::vector<float> vSumElement;
  std::vector<uint32_t> DetIds_ = getDetIds();
  int pos = getPosition(elementName);
   
  if (pos !=-1)
    {
      for (unsigned int i=0; i<DetIds_.size(); i++){
	const SiStripSummary::Range range = getRange(DetIds_.at(i));
	vSumElement.push_back(*((range.first)+pos));}
    }
   
  return vSumElement;
}




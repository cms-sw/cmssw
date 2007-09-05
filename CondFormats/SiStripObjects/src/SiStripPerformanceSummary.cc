#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"

//---- default constructor / destructor
SiStripPerformanceSummary::SiStripPerformanceSummary(): runNr_(0), timeValue_(0){}
SiStripPerformanceSummary::~SiStripPerformanceSummary(){ edm::LogInfo("Destructor")<<" SiStripPerformanceSummary destructor called."; }

//---- copy constructor
SiStripPerformanceSummary::SiStripPerformanceSummary(const SiStripPerformanceSummary& rhs){
  runNr_ = rhs.getRunNr();
  timeValue_ = rhs.getTimeValue();
  vDetSummary_ = rhs.getWholeSummary();
}

//---- set all summary values for one detid
std::pair<bool, std::vector<SiStripPerformanceSummary::DetSummary>::iterator> SiStripPerformanceSummary::setDet(const uint32_t input_detid, const std::vector<float>& input_values){
  std::vector<DetSummary>::iterator ivDet = vDetSummary_.end();
  // return false and end() if input vector not appropriate
  if(input_values.size() != kDetSummarySize) {
   edm::LogError("FillError")<<"wrong input size "<<input_values.size()<<". Can only add "<<kDetSummarySize<<" values. Not adding to SiStripPerformanceSummary";
    return std::make_pair(false, ivDet);
  }
  // return false and the old iterator if detid already exists
  ivDet = std::lower_bound(vDetSummary_.begin(),vDetSummary_.end(),input_detid,SiStripPerformanceSummary::StrictWeakOrdering());
  if (ivDet!=vDetSummary_.end() && ivDet->detId==input_detid){
    return std::make_pair(false, ivDet); // Already exists, not adding
  }
  // create detector summary for the input_detid, return true and the new iterator
  DetSummary detSummary;
  detSummary.detId=input_detid;
  detSummary.performanceValues = input_values;
  // put at the position-1 returned by the StrictWeakOrdering
  return std::make_pair(true, vDetSummary_.insert(ivDet, detSummary));
}

//---- initialize summary values of one detid to default nonsense
std::pair<bool, std::vector<SiStripPerformanceSummary::DetSummary>::iterator> SiStripPerformanceSummary::initDet(const uint32_t input_detid){ // initialize with defaults
  std::vector<float> input_values; for(int i = 0; i<kDetSummarySize; ++i) input_values.push_back(kNonsenseValue);
  return  setDet(input_detid, input_values);
}

//---- set two summary values of one detid
bool SiStripPerformanceSummary::setTwoValues(uint32_t input_detid, float val1, float val2, int index1, int index2){
  if(index1>kDetSummarySize || index2>kDetSummarySize){
   edm::LogError("SetError")<<" Could not set values for such indeces index1="<<index1<<" index2="<<index2<<" Maximum index is "<<kDetSummarySize;
   return false;
  }
  std::pair<bool, std::vector<DetSummary>::iterator> init_result = initDet(input_detid);
  if (init_result.first == true || init_result.second != vDetSummary_.end() ){ // new entry was created or existed before
    init_result.second->performanceValues[index1] = val1;
    init_result.second->performanceValues[index2] = val2;
    return true;
  }else{
    edm::LogError("SetError")<<" Could not set values, new entry could not be created for detid="<<input_detid;
    return false;
  }
  return true;
}

//---- set one summary value of one detid
bool SiStripPerformanceSummary::setOneValue(uint32_t input_detid, float val1, int index1){
  if(index1>kDetSummarySize){
   edm::LogError("SetError")<<" Could not set values for such index index1="<<index1<<" Maximum index is "<<kDetSummarySize;
   return false;
  }
  std::pair<bool, std::vector<DetSummary>::iterator> init_result = initDet(input_detid);
  if (init_result.first == true || init_result.second != vDetSummary_.end() ){ // new entry was created or existed before
    init_result.second->performanceValues[index1] = val1;
    return true;
  }else{
    edm::LogError("SetError")<<" Could not set values, new entry could not be created for detid="<<input_detid;
    return false;
  }
  return true;
}

//---- get one summary value of one detid
float SiStripPerformanceSummary::getOneValue(uint32_t input_detid, int index1){
  if(index1>kDetSummarySize){
    edm::LogError("GetError")<<" Could not get values for such index index1="<<index1<<" Maximum index is "<<kDetSummarySize;
    return kNonsenseValue;
  }
  std::vector<float> voutput; voutput.clear();
  getSummary(input_detid, voutput);
  if(voutput.size()==kDetSummarySize){
     return voutput[index1];
  }else{
    return kNonsenseValue;
  }
}

//---- add to input vector DetIds that have performance summary
void SiStripPerformanceSummary::getDetIds(std::vector<uint32_t>& vdetids) const {
  std::vector<DetSummary>::const_iterator begin = vDetSummary_.begin();
  std::vector<DetSummary>::const_iterator end   = vDetSummary_.end();
  for (std::vector<DetSummary>::const_iterator perf=begin; perf != end; ++perf) {
    vdetids.push_back(perf->detId);
  }
}

//---- print number of summaries
void SiStripPerformanceSummary::print() const{
  edm::LogInfo("print")<<"Nr. of elements in SiStripPerformanceSummary object is "<<  vDetSummary_.size()<<" RunNr="<<runNr_<<" timeValue="<<timeValue_<<std::endl;
}

//---- print all summary corresponding to one Detid
void SiStripPerformanceSummary::print(const uint32_t input_detid) const{
  DetSummary dummy; dummy.detId = input_detid;
  std::vector<DetSummary>::const_iterator ivDet = std::find_if(vDetSummary_.begin(),vDetSummary_.end(), MatchDetSummaryDetId(input_detid));
  if( ivDet==vDetSummary_.end() ){
    edm::LogError("print")<<"Cannot find any DetSummary for DetId="<<input_detid;
  }else{
    edm::LogInfo("print")<<"Input detid="<<input_detid<<"  DetSummary for DetId="<<ivDet->detId;
    print(ivDet->performanceValues);
  }
}

//---- return summary corresponding to one Detid
void SiStripPerformanceSummary::getSummary(const uint32_t input_detid, std::vector<float>& voutput) const{
  DetSummary dummy; dummy.detId = input_detid;
  std::vector<DetSummary>::const_iterator ivDet = std::find_if(vDetSummary_.begin(),vDetSummary_.end(), MatchDetSummaryDetId(input_detid));
  if( ivDet==vDetSummary_.end() ){
    edm::LogError("get")<<"Cannot find any DetSummary for DetId="<<input_detid;
  }else{
    voutput = ivDet->performanceValues;
  }
}

//---- print all summaries
void SiStripPerformanceSummary::printall() const{
  print();
  std::vector<DetSummary>::const_iterator begin = vDetSummary_.begin();
  std::vector<DetSummary>::const_iterator end   = vDetSummary_.end();
  for (std::vector<DetSummary>::const_iterator perf=begin; perf != end; ++perf) {
    std::cout<<" detid = "<<perf->detId;
    print(perf->performanceValues);
  }
}

//---- print a vector of floats
void SiStripPerformanceSummary::print(const std::vector<float>& pvec) const{
 for(std::vector<float>::const_iterator ip = pvec.begin(); ip != pvec.end(); ++ip) std::cout<<" "<<*ip;
 std::cout<<std::endl;
}


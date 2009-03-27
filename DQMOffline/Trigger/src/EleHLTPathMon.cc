#include "DQMOffline/Trigger/interface/EleHLTPathMon.h"



EleHLTPathMon::~EleHLTPathMon()
{
  for(size_t i=0;i<filters_.size();i++) delete filters_[i];
}

//a design decision to take a minor efficiency hit at object contruction to ensure that the vector is always sorted
//the alternative is to rely on the user (ha!) or a bool which is contiously checked to ensure the vector is sorted
void EleHLTPathMon::addFilter(const std::string& filterName)
{
  filters_.push_back(new EleHLTFilterMon(filterName,TrigCodes::getCode(filterName.c_str())));
  std::sort(filters_.begin(),filters_.end(),EleHLTFilterMon::ptrLess<EleHLTFilterMon>()); 
}


void EleHLTPathMon::addFilters(const std::vector<std::string>& names)
{
  for(size_t i=0;i<names.size();i++) addFilter(pathName_+names[i]);

}

//void EleHLTPathMon::setStdFilters()
//{
  //addFilter(pathName_+"L1MatchFilterRegional");
  //addFilter(pathName_+"EtFilter");
  //addFilter(pathName_+"HcalIsolFilter");
  //addFilter(pathName_+"PixelMatchFilter");
  //addFilter(pathName_+"EoverpFilter");
  //addFilter(pathName_+"HOneOEMinusOneOPFilter");
  //addFilter(pathName_+"TrackIsolFilter");
  //}



void EleHLTPathMon::fill(const EgHLTOffData& evtData,float weight)
{
  for(size_t filterNr=0;filterNr<filters_.size();filterNr++){ 
    // edm::LogInfo("EleHLTPathMon") << "fill filter "<<filterNr<<" / "<<filters_.size();
    filters_[filterNr]->fill(evtData,weight);
    //edm::LogInfo("EleHLTPathMon") << "filled filter "<<filterNr<<" / "<<filters_.size();
  }
}

std::vector<std::string> EleHLTPathMon::getFilterNames()const
{
  std::vector<std::string> names;
  for(size_t filterNr=0;filterNr<filters_.size();filterNr++){
    names.push_back(filters_[filterNr]->filterName());
  }
  return names;
}

#include "CondFormats/RunInfo/interface/HLTScaler.h"

lumi::HLTScaler::HLTScaler(){
  m_hltinfo.reserve(100);//hardcoded guess
}
size_t
lumi::HLTScaler::nHLTtrigger()const{
  return m_hltinfo.size();
}
lumi::HLTIterator 
lumi::HLTScaler::hltBegin()const{
  return m_hltinfo.begin();
}
lumi::HLTIterator
lumi::HLTScaler::hltEnd()const{
  return m_hltinfo.end();
}
void 
lumi::HLTScaler::setHLTData(const std::vector<lumi::HLTInfo>& hltdetail){
  std::copy(hltdetail.begin(),hltdetail.end(),std::back_inserter(m_hltinfo));
}


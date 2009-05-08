#include "CondFormats/RunInfo/interface/HLTScaler.h"

lumi::HLTScaler::HLTScaler(){
  m_hltinfo.reserve(100);//a guess
}
size_t
lumi::HLTScaler::nHLTPath()const{
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
lumi::HLTInfo 
lumi::HLTScaler::getHLTInfo( const std::string& pathname )const{
  for( std::vector< std::pair<std::string, lumi::HLTInfo> >::const_iterator it=m_hltinfo.begin();it!=m_hltinfo.end(); ++it){
    if( (*it).first==pathname ) return (*it).second;
  }
  return lumi::HLTNULL;
}
void 
lumi::HLTScaler::setHLTData(const std::vector< std::pair<std::string,lumi::HLTInfo> >& hltdetail){
  std::copy(hltdetail.begin(),hltdetail.end(),std::back_inserter(m_hltinfo));
}


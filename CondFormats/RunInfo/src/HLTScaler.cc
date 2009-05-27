#include "CondFormats/RunInfo/interface/HLTScaler.h"

lumi::HLTScaler::HLTScaler(): m_run(-99),m_lsnumber(-99){
  m_hltinfo.reserve(100);//a guess
}
int
lumi::HLTScaler::lumisectionNumber()const{
  return m_lsnumber;
}
int
lumi::HLTScaler::runNumber()const{
  return m_run;
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
bool 
lumi::HLTScaler::isNullData()const{
  return (m_lsnumber==-99);
}
void
lumi::HLTScaler::setHLTNULL(){
  m_run=-99;
  m_lsnumber=-99;
}
void 
lumi::HLTScaler::setHLTData(edm::LuminosityBlockID lumiid, 
			    const std::vector< lumi::HLTInfo >& hltdetail){
  m_run=lumiid.run();
  m_lsnumber=lumiid.luminosityBlock();
  std::copy(hltdetail.begin(),hltdetail.end(),std::back_inserter(m_hltinfo));
}

void
lumi::HLTScaler::print( std::ostream& s ) const{
  s<<"run number :"<<m_run <<", ";
  s<<"lumi section number : "<<m_lsnumber<<"\n";
  for( std::vector<lumi::HLTInfo>::const_iterator it=m_hltinfo.begin(); it!=m_hltinfo.end(); ++it ){
    s<<it->pathname<<": "<<it->inputcount<<": "<<it->acceptcount<<": "<<it->prescalefactor<<"\n";
  }
  s<<std::endl;
}

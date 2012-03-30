#include "RecoLuminosity/LumiProducer/interface/DIPLumiDetail.h"

#include <iomanip>
#include <ostream>
#include <iostream>

DIPLumiDetail::DIPLumiDetail():m_lumiValues(3564){
}
float 
DIPLumiDetail::lumiValue(unsigned int bx) const {
  return m_lumiValues[bx];
}
DIPLumiDetail::ValueRange
DIPLumiDetail::lumiValues() const {
  return ValueRange(m_lumiValues.begin(),m_lumiValues.end());
}
void
DIPLumiDetail::filldata(std::vector<float>& lumivalues){
  lumivalues.swap(m_lumiValues);
}
void
DIPLumiDetail::fillbxdata(unsigned int bxidx,float bxvalue){
  m_lumiValues[bxidx]=bxvalue;
}

std::ostream& operator<<(std::ostream& s, DIPLumiDetail const& diplumiDetail) {
  s << "\nDumping DIPLumiDetail\n";
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout.setf(std::ios::showpoint);
  std::vector<float>::const_iterator lumivalueIt= diplumiDetail.lumiValues().first;
  std::vector<float>::const_iterator lumivalueEnd = diplumiDetail.lumiValues().second;
  for(unsigned int i=0; lumivalueIt!=lumivalueEnd;++lumivalueIt,++i){
    s<<std::setw(10)<<" bunch = "<<i<<" bunchlumi = "<<*lumivalueIt << "\n";
  }
  s<<"\n";
  return s;
}

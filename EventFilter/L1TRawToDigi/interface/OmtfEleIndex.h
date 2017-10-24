#ifndef EventFilter_L1TRawToDigi_Omtf_EleIndex_H
#define EventFilter_L1TRawToDigi_Omtf_EleIndex_H

#include <cstdint>
#include <string>
#include <ostream>

namespace omtf {
class EleIndex {
public:
  EleIndex() : packed_(0) {}
  EleIndex(const std::string & board, unsigned int link) {
    unsigned int fed = 0;
    if (board.substr(4,1)=="n") fed = 1380; else if (board.substr(4,1)=="p") fed = 1381;
    unsigned int amc = std::stoi( board.substr(5,1) );
    packed_ = fed*1000+amc*100+link;
  }
  EleIndex(unsigned int fed, unsigned int amc, unsigned int link) { packed_ = fed*1000+amc*100+link; }
  unsigned int fed() const  { return packed_/1000; }
  unsigned int amc() const  { return ( (packed_ /100) %10); }
  unsigned int link() const { return packed_ % 100; }
  friend std::ostream & operator<< (std::ostream &out, const EleIndex &o) {
    out << "OMTF";
    if (o.fed()==1380) out <<"n";
    if (o.fed()==1381) out <<"p";
    out << o.amc();
    out <<" (fed: "<<o.fed()<<"), ln: " << o.link();
    return out;
  }
  inline bool operator< (const EleIndex& o) const { return this->packed_ < o.packed_; }

private:
  uint32_t packed_;

};

} //namespace imtf
#endif


#include "DataFormats/L1Trigger/interface/L1DataEmulDigi.h"
#include <iomanip>

L1DataEmulDigi::L1DataEmulDigi() {
  m_sid = -1; m_cid = -1;
  m_location[0]=99; m_location[1]=99; m_location[2]=99;
  m_type = -1; 
  std::fill(m_data,m_data+sizeof(m_data)/sizeof(m_data[0]),0);
  std::fill(m_rank,m_rank+sizeof(m_rank)/sizeof(m_rank[0]),0.);
}

L1DataEmulDigi::L1DataEmulDigi( int sid, int cid, double x1, double x2, double x3, int n)
{
  m_sid = sid; m_cid = cid;
  m_location[0]=x1; m_location[1]=x2; m_location[2]=x3;
  m_type = n; 
  std::fill(m_data,m_data+sizeof(m_data)/sizeof(m_data[0]),0);
  std::fill(m_rank,m_rank+sizeof(m_rank)/sizeof(m_rank[0]),0.);
}

L1DataEmulDigi::~L1DataEmulDigi() {}

std::ostream& operator<<(std::ostream& s, const L1DataEmulDigi& de) {
    unsigned word[2];
    float rankarr[2];
    de.data(word);
    de.rank(rankarr);
    s << "DEdigi"
      << " subsystem: "  << std::setw(2) << de.sid()
      << " (cid."        << std::setw(2) << de.cid() << ")"
      << " location: " 
      << "(" << std::setw(3) << std::setprecision(4) << de.x1() 
      << "," << std::setw(3) << std::setprecision(4) << de.x2() 
      << "," << std::setw(3) << std::setprecision(4) << de.x3() << ")"
      << " type: " << de.type()
      << std::hex
      << " dword:0x" << std::setw(8)<< word[0]
      << " eword:0x" << std::setw(8)<< word[1]
      << std::dec
      << " rank:"
      << "(" << std::setw(3) << std::setprecision(4) << rankarr[0] 
      << "," << std::setw(3) << std::setprecision(4) << rankarr[1] << ")";
    return s;
}


#include "DataFormats/L1Trigger/interface/L1DataEmulDigi.h"
#include <iomanip>

bool L1DataEmulDigi::empty() const {
  if(m_sid == m_null  || m_cid == m_null ) 
    return true;
  bool val = true;
  for(int i=0; i<2; i++) 
    val &= ( m_location[i]==m_null );
  return val;
}

int L1DataEmulDigi::reset() {
  m_null = -99;
  m_sid = m_null; 
  m_cid = m_null;
  for(int i=0; i<3; i++) 
    m_location[i]=m_null;
  m_type = m_null; 
  std::fill(m_data,m_data+sizeof(m_data)/sizeof(m_data[0]),0);
  std::fill(m_rank,m_rank+sizeof(m_rank)/sizeof(m_rank[0]),m_null);
  return m_null;
}

L1DataEmulDigi::L1DataEmulDigi() {
  reset();
}

L1DataEmulDigi::L1DataEmulDigi( int sid, int cid, double x1, double x2, double x3, int n) {
  reset();
  m_sid = sid; m_cid = cid;
  m_location[0]=x1; m_location[1]=x2; m_location[2]=x3;
  m_type = n; 
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
      << "(" << std::setw(5) << std::setprecision(2) << de.x1() 
      << "," << std::setw(5) << std::setprecision(2) << de.x2() 
      << "," << std::setw(5) << std::setprecision(2) << de.x3() << ")"
      << " type: " << de.type()
      << std::hex << std::setfill('0')
      << " dword:0x" << std::setw(8)<< word[0]
      << " eword:0x" << std::setw(8)<< word[1]
      << std::dec << std::setfill(' ')
      << " rank:"
      << "(" << std::setw(5) << std::setprecision(2) << rankarr[0] 
      << "," << std::setw(5) << std::setprecision(2) << rankarr[1] << ")";
    return s;
}


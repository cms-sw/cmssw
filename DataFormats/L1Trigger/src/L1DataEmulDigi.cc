#include "DataFormats/L1Trigger/interface/L1DataEmulDigi.h"
#include <iomanip>

bool L1DataEmulDigi::empty() const {
  if (m_sid == m_null || m_cid == m_null)
    return true;
  bool val = true;
  for (int i = 0; i < 2; i++)
    val &= (m_location[i] == m_null);
  return val;
}

int L1DataEmulDigi::reset() {
  m_null = -99;
  m_sid = m_null;
  m_cid = m_null;
  for (int i = 0; i < 3; i++)
    m_location[i] = m_null;
  m_type = m_null;
  std::fill(m_data, m_data + sizeof(m_data) / sizeof(m_data[0]), 0);
  std::fill(m_rank, m_rank + sizeof(m_rank) / sizeof(m_rank[0]), m_null);
  L1MonitorDigi def;
  m_DEpair[0] = def;
  m_DEpair[1] = def;
  return m_null;
}

L1DataEmulDigi::L1DataEmulDigi() { reset(); }

L1DataEmulDigi::L1DataEmulDigi(int sid, int cid, double x1, double x2, double x3, int n) {
  reset();
  m_sid = sid;
  m_cid = cid;
  m_location[0] = x1;
  m_location[1] = x2;
  m_location[2] = x3;
  m_type = n;
}

L1DataEmulDigi::L1DataEmulDigi(int sid,
                               int cid,
                               double x1,
                               double x2,
                               double x3,
                               int n,
                               unsigned int dw,
                               unsigned int ew,
                               float dr,
                               float er,
                               const L1MonitorDigi& dm,
                               const L1MonitorDigi& em) {
  reset();
  m_sid = sid;
  m_cid = cid;
  m_location[0] = x1;
  m_location[1] = x2;
  m_location[2] = x3;
  m_type = n;
  m_data[0] = dw;
  m_data[1] = ew;
  m_rank[0] = dr;
  m_rank[1] = er;
  m_DEpair[0] = dm;
  m_DEpair[1] = em;
}

L1DataEmulDigi::~L1DataEmulDigi() {}

std::ostream& operator<<(std::ostream& s, const L1DataEmulDigi& de) {
  unsigned word[2];
  float rankarr[2];
  de.data(word);
  de.rank(rankarr);
  s << "DEdigi"
    << " subsystem: " << std::setw(2) << de.sid() << " (cid." << std::setw(2) << de.cid() << ")"
    << " location: "
    << "(" << std::setw(5) << std::setprecision(2) << de.x1() << "," << std::setw(5) << std::setprecision(2) << de.x2()
    << "," << std::setw(5) << std::setprecision(2) << de.x3() << ")"
    << " type: " << de.type() << std::hex << std::setfill('0') << " dword:0x" << std::setw(8) << word[0] << " eword:0x"
    << std::setw(8) << word[1] << std::dec << std::setfill(' ') << " rank:"
    << "(" << std::setw(5) << std::setprecision(2) << rankarr[0] << "," << std::setw(5) << std::setprecision(2)
    << rankarr[1] << ")";
  return s;
}

GltDEDigi::GltDEDigi() { this->reset(); }

void GltDEDigi::reset() {
  const int w64 = 64;
  for (int j = 0; j < 2; j++) {
    globalDBit[j] = false;
    gltDecBits[j].reserve(w64 * 2);
    gltTchBits[j].reserve(w64);
    for (int i = 0; i < w64; i++) {
      gltDecBits[j][i] = false;
      gltDecBits[j][i + w64] = false;
      gltTchBits[j][i] = false;
    }
  }
}

GltDEDigi::GltDEDigi(bool glbit[], GltBits dbits[], GltBits tbits[]) { this->set(glbit, dbits, tbits); }

void GltDEDigi::set(bool glbit[], GltBits dbits[], GltBits tbits[]) {
  for (int i = 0; i < 2; i++) {
    globalDBit[i] = glbit[i];
    gltDecBits[i] = dbits[i];
    gltTchBits[i] = tbits[i];
  }
}

std::ostream& operator<<(std::ostream& s, const GltDEDigi& glt) {
  GltDEDigi::GltBits dbits[2], tbits[2];
  bool glbit[2];
  for (int i = 0; i < 2; i++) {
    glbit[i] = glt.globalDBit[i];
    dbits[i] = glt.gltDecBits[i];
    tbits[i] = glt.gltTchBits[i];
  }
  s << "GT DEdigi"
    << " decision: " << glbit[0];
  if (glbit[0] != glbit[1])
    s << "(data), " << glbit[1] << "(emul)";
  s << "\n data dec-word: ";
  for (GltDEDigi::GltBits::const_iterator i = dbits[0].begin(); i != dbits[0].end(); i++)
    s << *i;
  s << "\n emul dec-word: ";
  for (GltDEDigi::GltBits::const_iterator i = dbits[1].begin(); i != dbits[1].end(); i++)
    s << *i;
  s << "\n data techical: ";
  for (GltDEDigi::GltBits::const_iterator i = tbits[0].begin(); i != tbits[0].end(); i++)
    s << *i;
  s << "\n emul technical: ";
  for (GltDEDigi::GltBits::const_iterator i = tbits[1].begin(); i != tbits[1].end(); i++)
    s << *i;
  return s;
}

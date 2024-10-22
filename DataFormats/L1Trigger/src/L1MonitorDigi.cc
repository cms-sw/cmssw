#include "DataFormats/L1Trigger/interface/L1MonitorDigi.h"
#include <iomanip>

bool L1MonitorDigi::empty() const {
  if (m_sid == m_null || m_cid == m_null || m_value == m_null)
    return true;
  return false;
}

unsigned L1MonitorDigi::reset() {
  m_null = 999;
  m_sid = m_null;
  m_cid = m_null;
  for (int i = 0; i < 3; i++)
    m_location[i] = m_null;
  m_data = 0;
  m_value = m_null;
  return m_null;
}

L1MonitorDigi::L1MonitorDigi() { reset(); }

L1MonitorDigi::L1MonitorDigi(
    unsigned sid, unsigned cid, unsigned x1, unsigned x2, unsigned x3, unsigned value, unsigned data) {
  reset();
  m_sid = sid;
  m_cid = cid;
  m_location[0] = x1;
  m_location[1] = x2;
  m_location[2] = x3;
  m_value = value;
  m_data = data;
}

L1MonitorDigi::~L1MonitorDigi() {}

std::ostream& operator<<(std::ostream& s, const L1MonitorDigi& mon) {
  s << "L1Mon "
    << " system: " << std::setw(2) << mon.sid() << " (cid." << std::setw(2) << mon.cid() << ")" << std::hex
    << std::setfill('0') << " location: "
    << "(" << std::setw(5) << std::setprecision(2) << mon.x1() << "," << std::setw(5) << std::setprecision(2)
    << mon.x2() << "," << std::setw(5) << std::setprecision(2) << mon.x3() << ")"
    << " value: " << std::setw(5) << std::setprecision(2) << mon.value() << " word: " << std::setw(8) << mon.raw()
    << std::dec << std::setfill(' ');
  return s;
}

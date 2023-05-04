#include "DataFormats/L1Trigger/interface/L1DataEmulRecord.h"

L1DataEmulRecord::L1DataEmulRecord() : deAgree(false), deGlt() {
  for (int i = 0; i < DEnsys; i++) {
    deMatch[i] = false;
    deSysCompared[i] = false;
    for (int j = 0; j < 2; j++)
      deNCand[i][j] = 0;
  }
  deColl.clear();
}

L1DataEmulRecord::L1DataEmulRecord(bool evt_match,
                                   bool sys_comp[DEnsys],
                                   bool sys_match[DEnsys],
                                   int nCand[DEnsys][2],
                                   const L1DEDigiCollection& coll,
                                   const GltDEDigi& glt)
    : deAgree(evt_match), deGlt(glt) {
  for (int i = 0; i < DEnsys; i++) {
    deMatch[i] = sys_match[i];
    deSysCompared[i] = sys_comp[i];
    for (int j = 0; j < 2; j++)
      deNCand[i][j] = nCand[i][j];
  }
  deColl = coll;
}

L1DataEmulRecord::L1DataEmulRecord(bool evt_match,
                                   std::array<bool, DEnsys> const& sys_comp,
                                   std::array<bool, DEnsys> const& sys_match,
                                   std::array<std::array<int, 2>, DEnsys> const& nCand,
                                   const L1DEDigiCollection& coll,
                                   const GltDEDigi& glt)
    : deAgree(evt_match), deGlt(glt) {
  for (int i = 0; i < DEnsys; i++) {
    deMatch[i] = sys_match[i];
    deSysCompared[i] = sys_comp[i];
    for (int j = 0; j < 2; j++)
      deNCand[i][j] = nCand[i][j];
  }
  deColl = coll;
}

L1DataEmulRecord::~L1DataEmulRecord() {}

void L1DataEmulRecord::get_status(bool result[]) const {
  for (int i = 0; i < DEnsys; i++)
    result[i] = deMatch[i];
}

void L1DataEmulRecord::set_status(bool result) { deAgree = result; }

void L1DataEmulRecord::set_status(const bool result[]) {
  for (int i = 0; i < DEnsys; i++)
    deMatch[i] = result[i];
}

std::ostream& operator<<(std::ostream& s, const L1DataEmulRecord& cand) {
  s << "L1DataEmulRecord  d|e status: " << (cand.get_status() ? "agree" : "disagree");
  s << "\n\tsys compd? ";
  for (int i = 0; i < L1DataEmulRecord::DEnsys; i++)
    s << cand.get_isComp(i);
  s << "\n\tsys match? ";
  for (int i = 0; i < L1DataEmulRecord::DEnsys; i++)
    s << cand.get_status(i);
  s << "\n\tndata: ";
  for (int i = 0; i < L1DataEmulRecord::DEnsys; i++)
    s << cand.getNCand(i, 0) << " ";
  s << "\n\tnemul: ";
  for (int i = 0; i < L1DataEmulRecord::DEnsys; i++)
    s << cand.getNCand(i, 1) << " ";
  s << "\n\tdigis  size:" << (cand.getColl()).size();
  s << std::flush;
  L1DataEmulRecord::L1DEDigiCollection::const_iterator it;
  std::vector<L1DataEmulDigi> dgcoll = cand.getColl();
  for (it = dgcoll.begin(); it != dgcoll.end(); it++)
    s << "\n\t" << *it;
  s << cand.getGlt();
  return s;
}

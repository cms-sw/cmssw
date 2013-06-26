#ifndef DataFormats_L1DataEmulRecord_h
#define DataFormats_L1DataEmulRecord_h

/*\class L1DataEmulRecord
 *\description L1 trigger data|emulation event record
 *\author Nuno Leonardo (CERN)
 *\date 07.06
 */

#include <ostream>
#include <vector>

#include "DataFormats/L1Trigger/interface/L1DataEmulDigi.h"

class L1DataEmulRecord {

 public:

  static const int DEnsys = 12; 
  typedef std::vector<L1DataEmulDigi> L1DEDigiCollection;

  L1DataEmulRecord();
  L1DataEmulRecord(bool evt_match, bool sys_comp[DEnsys], bool sys_match[DEnsys], 
		   int nCand[DEnsys][2], const L1DEDigiCollection&, GltDEDigi); 
  ~L1DataEmulRecord();
  
  bool get_status() const { return deAgree; }
  bool get_status(int s) const { return deMatch[s];}
  void get_status(bool result[]) const;
  L1DEDigiCollection getColl() const {return deColl;}
  GltDEDigi getGlt() const {return deGlt;}
  int getNCand(int i, int j) const {return deNCand[i][j];}
  bool get_isComp(int i) const {return deSysCompared[i];}

  void set_status(const bool result);
  void set_status(const bool result[]); 
  void setColl(const L1DEDigiCollection& col) {deColl = col;}
  void setGlt(GltDEDigi glt) {deGlt = glt;}
  
  bool empty() const {return deColl.size()==0;}

 private:

  bool deAgree; 
  bool deSysCompared[DEnsys];  
  bool deMatch[DEnsys];
  int deNCand[DEnsys][2];
  L1DEDigiCollection deColl;
  GltDEDigi deGlt;
      
};

std::ostream& operator<<(std::ostream&, const L1DataEmulRecord&);

#endif

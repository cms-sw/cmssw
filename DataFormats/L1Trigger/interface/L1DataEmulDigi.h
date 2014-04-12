#ifndef DataFormats_L1DataEmulDigi_h
#define DataFormats_L1DataEmulDigi_h

/*\class L1DataEmulDigi
 *\description L1 trigger data|emulation digi
 *\author Nuno Leonardo (CERN)
 *\date 07.07
 */

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/L1Trigger/interface/L1MonitorDigi.h"

class L1DataEmulDigi {

 public:

  L1DataEmulDigi();
  L1DataEmulDigi(int sid, int cid, double x1, double x2, double x3, int n);
  L1DataEmulDigi(int sid, int cid, double x1, double x2, double x3, int n,
		 unsigned int dw, unsigned int ew, float dr, float er, 
		 const L1MonitorDigi& dm, const L1MonitorDigi& em); 
  ~L1DataEmulDigi();
  
  void setSid (int sid) {m_sid = sid;}
  void setCid (int cid) {m_cid = cid;}
  void setLocation (double x1, double x2, double x3) 
    { m_location[0]=x1; m_location[1]=x2; m_location[2]=x3;}
  void setType (int n) {m_type = n;}
  void setData(unsigned int d, unsigned int e) {m_data[0]=d; m_data[1]=e;}
  void setRank(float d, float e) {m_rank[0]=d; m_rank[1]=e;}
  void setDEpair(const L1MonitorDigi& d, const L1MonitorDigi& e) 
    {m_DEpair[0]=d; m_DEpair[1]=e; }

  int sid() const {return m_sid;}
  int cid() const {return m_cid;}
  double x1() const {return m_location[0];}
  double x2() const {return m_location[1];}
  double x3() const {return m_location[2];}
  int type() const {return m_type;}
  void data(unsigned int *d) const {d[0]=m_data[0]; d[1]=m_data[1];}
  void rank(float *r) const {r[0]=m_rank[0]; r[1]=m_rank[1];}
  L1MonitorDigi data() const {return m_DEpair[0];}
  L1MonitorDigi emul() const {return m_DEpair[1];}

  int reset();
  bool empty() const;

 private:

  int m_sid;
  int m_cid;
  double m_location[3];
  int m_type;
  unsigned int m_data[2];
  float m_rank[2];
  int m_null;
  L1MonitorDigi m_DEpair[2]; 

};

std::ostream& operator<<(std::ostream&, const L1DataEmulDigi&);

struct GltDEDigi {
  typedef std::vector<bool> GltBits;
  GltDEDigi();
  void reset();
  GltDEDigi(bool glbit[], GltBits dbits[], GltBits tbits[]);
  void set(bool glbit[], GltBits dbits[], GltBits tbits[]);
  bool globalDBit[2];
  GltBits gltDecBits[2], gltTchBits[2];
};

std::ostream& operator<<(std::ostream&, const GltDEDigi&);

#endif

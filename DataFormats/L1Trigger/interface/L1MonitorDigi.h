#ifndef DataFormats_L1Monitor_h
#define DataFormats_L1Monitor_h

/*\class L1MonitorDigi
 *\description L1 trigger generic digi for monitoring 
 *\author Nuno Leonardo (CERN)
 *\date 08.03
 */

#include <ostream>
#include <string>
#include <utility>

class L1MonitorDigi {

 public:

  L1MonitorDigi();
  L1MonitorDigi(unsigned sid, unsigned cid, unsigned x1, unsigned x2, 
		unsigned x3, unsigned value, unsigned data);
  ~L1MonitorDigi();
  
  void setSid(int sid) {m_sid = sid;}
  void setCid(int cid) {m_cid = cid;}
  void setLoc(unsigned x1, unsigned x2, unsigned x3) 
    { m_location[0]=x1; m_location[1]=x2; m_location[2]=x3;}
  void setRaw(unsigned raw) {m_data=raw;}
  void setValue(unsigned val) {m_value=val;}
  
  unsigned sid()   const {return m_sid;}
  unsigned cid()   const {return m_cid;}
  unsigned x1()    const {return m_location[0];}
  unsigned x2()    const {return m_location[1];}
  unsigned x3()    const {return m_location[2];}
  unsigned raw()   const {return m_data;}
  unsigned value() const {return m_value;}

  unsigned reset();
  bool empty() const;

 private:

  unsigned m_sid;
  unsigned m_cid;
  unsigned m_location[3];
  unsigned m_value;
  unsigned m_data;
  unsigned m_null;

};

std::ostream& operator<<(std::ostream&, const L1MonitorDigi&);

#endif

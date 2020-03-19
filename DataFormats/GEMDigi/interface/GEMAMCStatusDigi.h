#ifndef DataFormats_GEMDigi_GEMAMCStatusDigi_H
#define DataFormats_GEMDigi_GEMAMCStatusDigi_H

#include <cstdint>
#include <vector>

class GEMAMCStatusDigi {
public:
  explicit GEMAMCStatusDigi(const uint8_t AMCnum_,
                            const uint32_t L1A_,
                            const uint16_t BX_,
                            const uint32_t Dlength_,
                            const uint8_t FV_,
                            const uint8_t Rtype_,
                            const uint8_t Param1_,
                            const uint8_t Param2_,
                            const uint8_t Param3_,
                            const uint16_t Onum_,
                            const uint16_t BID_,
                            const uint32_t GEMDAV_,
                            const uint64_t Bstatus_,
                            const uint8_t GDcount_,
                            const uint8_t Tstate_,
                            const uint32_t ChamT_,
                            const uint8_t OOSG);

  GEMAMCStatusDigi() {}

  uint8_t AMCnum() const { return m_AMCnum; }     ///<Returns AMC number
  uint32_t L1A() const { return m_L1A; }          ///<Returns L1A number
  uint16_t BX() const { return m_BX; }            ///<Returns Bunch Crossing ID
  uint32_t Dlength() const { return m_Dlength; }  ///<Returns Data Length (Overall size of FED event fragment)

  uint8_t FV() const { return m_FV; }        ///<Returns Format Version
  uint8_t Rtype() const { return m_Rtype; }  ///<Returns Run Type
  uint8_t Param1() const { return m_Param1; }
  uint8_t Param2() const { return m_Param2; }
  uint8_t Param3() const { return m_Param3; }
  uint16_t Onum() const { return m_Onum; }  ///<Returns Orbit number
  uint16_t BID() const { return m_BID; }    ///<Returns Board ID

  uint32_t GEMDAV() const { return m_GEMDAV; }         ///<Returns GEM DAV list (which chambers have data)
  uint64_t Bstatus() const { return m_Bstatus; }       ///<Returns Buffer status
  int GDcount() const { return unsigned(m_GDcount); }  ///<Returns GEM DAV count (number of chamber blocks)
  uint8_t Tstate() const { return m_Tstate; }          ///<Returns TTS state

  uint32_t ChamT() const { return m_ChamT; }  ///<Return Chamber Timeout
  uint8_t OOSG() const { return m_OOSG; }     ///<Return OOS GLIB (if GLIB is out of sync)

  uint32_t CRC() const { return m_CRC; }
  uint8_t L1AT() const { return m_L1AT; }
  uint32_t DlengthT() const { return m_DlengthT; }

private:
  uint8_t m_AMCnum;
  uint32_t m_L1A;
  uint16_t m_BX;
  uint32_t m_Dlength;

  uint8_t m_FV;  ///<0000:4    Format Version:4
  uint8_t m_Rtype;
  uint8_t m_Param1;  ///<Run param1:8
  uint8_t m_Param2;  ///<Run param2:8
  uint8_t m_Param3;  ///<Run param3:8
  uint16_t m_Onum;   ///<Orbit number:16
  uint16_t m_BID;

  uint32_t m_GEMDAV;
  uint64_t m_Bstatus;
  uint8_t m_GDcount;
  uint8_t m_Tstate;

  uint32_t m_ChamT;
  uint8_t m_OOSG;

  uint32_t m_CRC;
  uint8_t m_L1AT;
  uint32_t m_DlengthT;
};
#endif

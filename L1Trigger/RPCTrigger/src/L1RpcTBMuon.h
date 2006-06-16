/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2005                                                      *
*                                                                              *
*******************************************************************************/
//---------------------------------------------------------------------------
#ifndef L1RpcTBMuonH
#define L1RpcTBMuonH
#include "L1Trigger/RPCTrigger/src/L1RpcMuon.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPacMuon.h"
//---------------------------------------------------------------------------
//output of the Pac (one LogCone),
//used in TBGhostBuster
//input and output of the TBGhostBuster

/** \class L1RpcTBMuon
  * Used in Ghoust Busters and sorters.
  * Has additionall filds: Killed, GBData, EtaAddress, PhiAddress need in these algorithms.
  * \author Karol Bunkowski (Warsaw)
  */


class L1RpcTBMuon: public L1RpcMuon {
public:
  ///Empty muon.
  L1RpcTBMuon(): L1RpcMuon() {
    Killed = false;

    GBData = 0;

    EtaAddress = 0;
    PhiAddress = 0;
  };

  L1RpcTBMuon(int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes):
    L1RpcMuon(ptCode, quality, sign, patternNum, firedPlanes) {
    Killed = false;

    GBData = 0;

    EtaAddress = 0;
    PhiAddress = 0;
  };

  L1RpcTBMuon(const L1RpcPacMuon& pacMuon):
    L1RpcMuon(pacMuon) {
//    L1RpcMuon(pacMuon.GetConeCrdnts(), pacMuon.GetPtCode(), pacMuon.GetQuality(), pacMuon.GetSign(), pacMuon.GetPatternNum()) {

    Killed = false;

    GBData = 0;

    EtaAddress = 0;
    PhiAddress = 0;
  };

  ///Combined quality and ptCode, 8 bits [7...5 Quality, 4...0 PtCode], used in GhoustBusters
  int GetCode() const {
    return (Quality<<5 | PtCode);
  };

  ///Sets combined code: 8 bits [7...5 Quality, 4...0 PtCode].
  void SetCode(int code) {
    Quality = (code & (3<<5))>>5;
    PtCode = code & 31;
  };
//-----Addres-------------------------------
  void SetPhiAddr(int phiAddr) {
    PhiAddress = phiAddr;
  }

  void SetSectorAddr(int sectorAddr){
    PhiAddress = PhiAddress | sectorAddr<<4;
  }

  void SetEtaAddr(int etaAddr) {
    EtaAddress = etaAddr;
  }

  /*
  //tower addres on TB (0-2 or 0-3)
  void SetTbTowAddr(int tbTower) {
    EtaAddress = EtaAddress | tbTower;
  } */

  /*
  void SetTbAddr(int tbNumber) {
    EtaAddress = EtaAddress | (tbNumber<<2);
  } */
  
  void SetAddress(int etaAddr, int phiAddr) {
    EtaAddress = etaAddr;
    PhiAddress = phiAddr;
  }

  void SetAddress(int tbNumber, int tbTower, int phiAddr) {
    EtaAddress = (tbNumber<<2) | tbTower;
    PhiAddress = phiAddr;
  }

  int GetEtaAddr() const {
    return EtaAddress;
  }
  /*
  int GetTbNumAddr() const {
    return ((EtaAddress & 60)>>2);
  }

  int GetTbTowAddr() const {
    return (EtaAddress & 3);
  } */

  int GetPhiAddr() const {
    return PhiAddress;
  }

  int GetSegmentAddr() const {
    return PhiAddress & 15;
  }

  int GetSectorAddr() const {
    return (PhiAddress & 0xF0)>>4;
  }

  int GetContinSegmAddr() const {
    return GetSectorAddr()*12 + GetSegmentAddr();
  }

  void SetCodeAndPhiAddr(int code, int phiAddr) {
    SetCode(code);
    PhiAddress = phiAddr;
  };

  void SetCodeAndEtaAddr(int code, int etaAddr) {
    SetCode(code);
    EtaAddress = etaAddr;
  };
//-----------------------------------------------------
  int GetGBData() const {
    return GBData;
  }

  std::string GetGBDataBitStr() const {
    std::string str = "00";
    if (GBData == 1)
      str = "01";
    else if (GBData == 2)
      str = "10";
    else if (GBData == 3)
      str = "11";
    return str;  
  }

  void SetGBDataKilledFirst() {
    GBData = GBData | 1;
  }

  void SetGBDataKilledLast() {
    GBData = GBData | 2;
  }

  bool GBDataKilledFirst() const {
    return (GBData & 1);
  }

  bool GBDataKilledLast() const {
    return (GBData & 2);
  }

  /** @param where - to bits meaning is differnt on diffrent places
   * values:
   * 	fsbOut - outputo of the Final Sorter Board - input of GMT	
   * 
   * */ 		
  unsigned int ToBits(std::string where) const;
  
  void FromBits(std::string where, unsigned int value);
  
  std::string BitsToString() const;

//------need to perform ghost - busting ------------
  void Kill() {
    Killed = true;
  }

  /** @return true = was non-empty muon and was killed
    * false = was not killed or is zero */
  bool WasKilled() const {
    if(PtCode > 0 && Killed)
      return true;
    else return false;
  };

  /** @return true = was no-zero muon and was not killed
    * false = is killed or is zero */
  bool IsLive() const {
    if(PtCode > 0 && !Killed)
      return true;
    else return false;
  };

  ///Used in sorting.
  struct TMuonMore : public std::less<L1RpcTBMuon> {
    bool operator() (const L1RpcTBMuon& muonL,
                                 const L1RpcTBMuon& muonR) const {
      /*
      if(muonL.GetCode() == muonR.GetCode() ) {
        if(muonL.GetEtaAddr() == muonR.GetEtaAddr() )
          return muonL.GetPhiAddr() < muonR.GetPhiAddr();
        return muonL.GetEtaAddr() < muonR.GetEtaAddr();
      }*/
      return muonL.GetCode() > muonR.GetCode();
    }
  };

private:
//------ hardware signals------------------------
  unsigned int EtaAddress;

  unsigned int PhiAddress;

  /** 2 bits,
    * 0 00 - this muon did not kill nothing on the sector edge
    * 1 01 - this muon killed muon from segment 0 (the "right" sector edge), or is in segment 0
    * 2 10 - this muon killed muon from segment 11 (the "left" sector edge), or is in segment 11
    * 3 11 - this muon killed muon from segment 0 and 11 */
  unsigned int GBData; 

//------- need to perform ghost - busting ---------
  bool Killed; //!< true means that muon was killed during GB
	
  class FSBOut {
  private:
  	static const int phiBitsCnt  = 8;  static const unsigned int phiBitsMask  = 0xff;
  	static const int ptBitsCnt   = 5;  static const unsigned int ptBitsMask   = 0x1f;
  	static const int qualBitsCnt = 3;  static const unsigned int qualBitsMask = 0x7;
  	static const int etaBitsCnt  = 6;  static const unsigned int etaBitsMask  = 0x3f;
  	static const int signBitsCnt = 1;  static const unsigned int signBitsMask = 0x1;
  public:	
  	static unsigned int toBits(const L1RpcTBMuon& muon);
  	static void fromBits(L1RpcTBMuon& muon, unsigned int value);
  };
  friend class FSBOut;

  class FSBIn {
  private:
  	static const int signBitsCnt = 1;  static const unsigned int signBitsMask = 0x1;
  	static const int ptBitsCnt   = 5;  static const unsigned int ptBitsMask   = 0x1f;
  	static const int qualBitsCnt = 3;  static const unsigned int qualBitsMask = 0x7;  	
  	static const int phiBitsCnt  = 7;  static const unsigned int phiBitsMask  = 0x7f;  	  	
  	static const int etaBitsCnt  = 6;  static const unsigned int etaBitsMask  = 0x3f;
  public:	  	
  	static unsigned int toBits(const L1RpcTBMuon& muon);
  	static void fromBits(L1RpcTBMuon& muon, unsigned int value);
  };
  friend class FSBIn;
  
};

typedef std::vector<L1RpcTBMuon> L1RpcTBMuonsVec;
typedef std::vector<L1RpcTBMuonsVec> L1RpcTBMuonsVec2;

#endif


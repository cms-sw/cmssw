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
  L1RpcTBMuon();

  L1RpcTBMuon(int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes);

  L1RpcTBMuon(const L1RpcPacMuon& pacMuon);

  int GetCode() const;

  void SetCode(int code);



  void SetPhiAddr(int phiAddr);

  void SetSectorAddr(int sectorAddr);

  void SetEtaAddr(int etaAddr);
  
  void SetAddress(int etaAddr, int phiAddr);

  void SetAddress(int tbNumber, int tbTower, int phiAddr);

  int GetEtaAddr() const;

  int GetPhiAddr() const;

  int GetSegmentAddr() const;

  int GetSectorAddr() const;

  int GetContinSegmAddr() const;

  void SetCodeAndPhiAddr(int code, int phiAddr);

  void SetCodeAndEtaAddr(int code, int etaAddr);

  int GetGBData() const;

  std::string GetGBDataBitStr() const;
  
  std::string printDebugInfo(int) const;

  std::string printExtDebugInfo(int, int, int) const;

  void SetGBDataKilledFirst();

  void SetGBDataKilledLast();

  bool GBDataKilledFirst() const;

  bool GBDataKilledLast() const;

  /** @param where - to bits meaning is differnt on diffrent places
   * values:
   * 	fsbOut - outputo of the Final Sorter Board - input of GMT	
   * 
   * */ 		
  unsigned int ToBits(std::string where) const;
  
  void FromBits(std::string where, unsigned int value);
  
  std::string BitsToString() const;

//------need to perform ghost - busting ------------
  void Kill();

  /** @return true = was non-empty muon and was killed
    * false = was not killed or is zero */
  bool WasKilled() const;

  /** @return true = was no-zero muon and was not killed
    * false = is killed or is zero */
  bool IsLive() const;
//aaa
  ///Used in sorting.
  struct TMuonMore : public std::less<L1RpcTBMuon> {
    bool operator() (const L1RpcTBMuon& muonL,
                     const L1RpcTBMuon& muonR) const {
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


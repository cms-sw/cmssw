#ifndef L1ScoutingRawData_SDSNumbering_h
#define L1ScoutingRawData_SDSNumbering_h

/**
  *
  * This class holds the Scouting Data Source (SDS)
  * numbering scheme for the Level 1 scouting system
  *
  */

class SDSNumbering {
public:
  static constexpr int lastSDSId() { return MAXSDSID; }

  static constexpr int NOT_A_SDSID = -1;
  static constexpr int MAXSDSID = 32;
  static constexpr int GmtSDSID = 1;
  static constexpr int CaloSDSID = 2;
  static constexpr int GtSDSID = 4;
  static constexpr int BmtfMinSDSID = 10;
  static constexpr int BmtfMaxSDSID = 21;
  static constexpr int CaloTCPMinSDSID = 22;
  static constexpr int CaloTCPMaxSDSID = 29;
  static constexpr int CaloTowerMinSDSID = 32;
  static constexpr int CaloTowerMaxSDSID = 32;
};

#endif  // L1ScoutingRawData_SDSNumbering_h

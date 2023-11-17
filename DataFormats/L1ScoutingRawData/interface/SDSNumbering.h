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

    enum {
      NOT_A_SDSID = -1,
      MAXSDSID = 32,
      GmtSDSID = 1,
      CaloSDSID = 2,
      GtSDSID = 4,
      BmtfMinSDSID = 10,
      BmtfMaxSDSID = 21
    };
};

#endif // L1ScoutingRawData_SDSNumbering_h
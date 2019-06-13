#ifndef MuonDetId_CSCTriggerNumbering_h
#define MuonDetId_CSCTriggerNumbering_h

/** \class CSCTriggerNumbering
 * 
 * Converts standard trigger labels to geometry labels.
 * "Standard" implies that the numbering is EXACTLY that of 
 * CMS NOTE: CMS IN 2000/004 v. 2.1 (Oct, 2005).
 * 
 * \warning EVERY INDEX COUNTS FROM ONE
 */

class CSCDetId;

class CSCTriggerNumbering {
public:
  CSCTriggerNumbering();
  ~CSCTriggerNumbering();

  /**
   * The following functions transform standard chamber labels into
   * their corresponding trigger labels.
   */

  /**
    * Return trigger-level sector id for an Endcap Muon chamber.
    *
    * This method encapsulates the information about which chambers
    * are in which sectors, and may need updating according to
    * hardware changes, or software chamber indexing.
    *
    * Station 1 has 3 rings of 10-degree chambers. <br>
    * Stations 2, 3, 4 have an inner ring of 20-degree chambers
    * and an outer ring of 10-degree chambers. <br>
    *
    * Sectors are 60 degree slices of a station, covering both rings. <br>
    * For Station 1, there are subsectors of 30 degrees: 9 10-degree
    * chambers (3 each from ME1/1, ME1/2, ME1/3.) <br>
    * 
    * The first sector starts at phi = 15 degrees so it matches Barrel Muon sectors.
    * We count from one not zero.
    *
    */
  static int triggerSectorFromLabels(int station, int ring, int chamber);
  static int triggerSectorFromLabels(CSCDetId id);

  /**
   * Return trigger-level sub sector id within a sector in station one.
   * 
   * Each station one sector has two 30 degree subsectors.
   * Again, we count from one, not zero. Zero is an allowed return value though.
   *  
   * A return value of zero means this station does not have subsectors.
   *
   */
  static int triggerSubSectorFromLabels(int station, int chamber);
  static int triggerSubSectorFromLabels(CSCDetId id);

  /**
    * Return trigger-level CSC id  within a sector for an Endcap Muon chamber.
    *
    * This id is an index within a sector such that the 3 inner ring chambers 
    * (20 degrees each) are 1, 2, 3 (increasing counterclockwise) and the 6 outer ring 
    * chambers (10 degrees each) are 4, 5, 6, 7, 8, 9 (again increasing counter-clockwise.) 
    *
    * This method knows which chambers are part of which sector and returns
    * the chamber label/index/identifier accordingly.
    * Beware that this information is liable to change according to hardware
    * and software changes.
    *
    */
  static int triggerCscIdFromLabels(int station, int ring, int chamber);
  static int triggerCscIdFromLabels(CSCDetId id);

  /**
   * The following functions transform trigger labels into their
   * corresponding standard chamber labels.
   */

  /**
   * \function ringFromTriggerLabels
   *
   * This function calculates the ring at which a given chamber resides.
   * Station 1: ring = [1,3]
   * Station 2-4: ring = [1,2]
   */
  static int ringFromTriggerLabels(int station, int triggerCSCID);

  /**
   * \function chamberFromTriggerLabels
   *
   * This function calculates the chamber number for a given set of
   * trigger labels.   
   */
  static int chamberFromTriggerLabels(int TriggerSector, int TriggerSubSector, int station, int TriggerCSCID);

  /**
   * \function sectorFromTriggerLabels
   *
   * Translates trigger sector and trigger subsector into the "real" sector number
   * For station 1 sector = [1,12]
   * For stations 2-4 sector = [1,6]
   */
  static int sectorFromTriggerLabels(int TriggerSector, int TriggerSubSector, int station);

  /**
   * Minimum and Maximum values for trigger specific labels.
   */

  static int maxTriggerCscId() { return MAX_CSCID; }
  static int minTriggerCscId() { return MIN_CSCID; }
  static int maxTriggerSectorId() { return MAX_TRIGSECTOR; }
  static int minTriggerSectorId() { return MIN_TRIGSECTOR; }
  static int maxTriggerSubSectorId() { return MAX_TRIGSUBSECTOR; }
  static int minTriggerSubSectorId() { return MIN_TRIGSUBSECTOR + 1; }

private:
  // Below are counts for trigger based labels.

  // Max counts for trigger labels.
  enum eTrigMaxNum { MAX_TRIGSECTOR = 6, MAX_CSCID = 9, MAX_TRIGSUBSECTOR = 2 };

  // Min counts for trigger labels. Again, we count from one.
  enum eTrigMinNum { MIN_TRIGSECTOR = 1, MIN_CSCID = 1, MIN_TRIGSUBSECTOR = 0 };
};

#endif

#ifndef MuonDetId_CSCDetId_h
#define MuonDetId_CSCDetId_h

/** \class CSCDetId
 * Identifier class for hierarchy of Endcap Muon detector components.
 *
 * Ported from MuEndDetectorId but now derived from DetId and updated accordingly.
 *
 * Allows access to hardware integer labels of the subcomponents
 * of the Muon Endcap CSC detector system.
 *
 * The STATIC member functions can be used to translate back and
 * forth between a layer/chamber 'rawId' and the %set of subdetector labels.
 *
 * \warning EVERY LABEL COUNTS FROM ONE NOT ZERO.
 *
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

class CSCDetId;

std::ostream& operator<<( std::ostream& os, const CSCDetId& id );

class CSCDetId : public DetId {

public:


  /// Default constructor; fills the common part in the base
  /// and leaves 0 in all other fields
  CSCDetId() : DetId(DetId::Muon, MuonSubdetId::CSC){}

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is CSC, otherwise an exception is thrown.
  CSCDetId(uint32_t id) : DetId(id) {}
  CSCDetId(DetId id) : DetId(id) {}


  /// Construct from fully qualified identifier.
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.<br>
  /// iendcap: 1=forward (+Z), 2=backward(-Z)
  CSCDetId( int iendcap, int istation, 
	    int iring, int ichamber, 
	    int ilayer = 0 ) : DetId(DetId::Muon, MuonSubdetId::CSC) {
     id_ |= init(iendcap, istation, iring, ichamber, ilayer);
  }

  /** Chamber CSCDetId from a Layer CSCDetId
   */
  CSCDetId chamberId() const {
    // build chamber id by removing layer bits
    return CSCDetId( id_ - layer() ) ; }

  /**
   * Return Layer label.
   *
   */
  int layer() const {
    return (id_ & MASK_LAYER); } 

  /**
   * Return Chamber label.
   *
   */
   int chamber() const {
     return (  (id_>>START_CHAMBER) & MASK_CHAMBER ); }

  /**
   * Return Ring label.
   *
   */
   int ring() const {
     if (((id_>>START_STATION) & MASK_STATION) == 1)
       return (  detIdToInt((id_>>START_RING) & MASK_RING )); 
     else
       return (((id_>>START_RING) & MASK_RING )); 
   }

  /**
   * Return Station label.
   *
   */
   int station() const {
     return (  (id_>>START_STATION) & MASK_STATION ); }

  /**
   * Return Endcap label. 1=forward (+Z); 2=backward (-Z)
   *
   */
   int endcap() const {
     return (  (id_>>START_ENDCAP) & MASK_ENDCAP ); }

   /**
    * What is the sign of global z?
    *
    */
   short int zendcap() const {
     return ( endcap()!=1 ? -1 : +1 );
   }

   /**
    * Chamber type (integer 1-10)
    */
   unsigned short iChamberType() const {
     return iChamberType( station(), ring() );
   }

   /**
    * Geometric channel no. from geometric strip no. - identical except for ME1a ganged strips
    *
    * Note that 'Geometric' means increasing number corresponds to increasing local x coordinate. 
    * \warning There is no attempt here to handle cabling or readout questions. 
    * If you need that look at CondFormats/CSCObjects/CSCChannelTranslator.
    */
   int channel( int istrip ) { 
     if ( ring()== 4 ) 
       // strips 1-48 mapped to channels 1-16:
       // 1+17+33->1, 2+18+34->2, .... 16+32+48->16
       return 1 + (istrip-1)%16; 
     else 
       return istrip; 
   }

  // static methods
  // Used when we need information about subdetector labels.

  /** 
   * Returns the unique integer 'rawId' which labels each CSC layer.
   *
   * The arguments are the integer labels for, respectively,  <br>
   * endcap, station, ring, chamber, layer.
   *
   * \warning The input int args are expected to be .ge. 1 and there
   * is no sanity-checking for their upper limits.
   *
   * \warning The returned integers are not necessarily consecutive, i.e.
   * there are gaps. This is to permit computational efficiency
   * starting from the component ids.
   *
   */
   static int rawIdMaker( int iendcap, int istation, int iring, 
               int ichamber, int ilayer ) {
     return ( (DetId::Muon&0xF)<<(DetId::kDetOffset) ) |            // set Muon flag
            ( (MuonSubdetId::CSC&0x7)<<(DetId::kSubdetOffset) ) |   // set CSC flag
               init(iendcap, istation, iring, ichamber, ilayer) ; } // set CSC id

   /**
    * Return Layer label for supplied CSCDetId index.
    *
    */
   static int layer( int index ) {
     return (index & MASK_LAYER); }

   /**
    * Return Chamber label for supplied CSCDetId index.
    *
    */
   static int chamber( int index ) {
     return (  (index>>START_CHAMBER) & MASK_CHAMBER ); }

   /**
    * Return Ring label for supplied CSCDetId index.
    *
    */
   static int ring( int index ) {
     if (((index>>START_STATION) & MASK_STATION) == 1)
       return (  detIdToInt((index>>START_RING) & MASK_RING )); 
     else
       return (( index>>START_RING) & MASK_RING ); 
   }

   /**
    * Return Station label for supplied CSCDetId index.
    *
    */
   static int station( int index ) {
     return (  (index>>START_STATION) & MASK_STATION ); }

   /**
    * Return Endcap label for supplied CSCDetId index.
    *
    */
   static int endcap( int index ) {
     return (  (index>>START_ENDCAP) & MASK_ENDCAP ); }

   /**
    * Return a unique integer 1-10 for a station, ring pair:
    *        1           for S = 1  and R=4 inner strip part of ME11 (ME1a)
    *      2,3,4 =  R+1  for S = 1  and R = 1,2,3 (ME11 means ME1b)
    *      5-10  = 2*S+R for S = 2,3,4 and R = 1,2
    */
   static unsigned short iChamberType( unsigned short istation, unsigned short iring );

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
   int triggerSector() const;

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
   int triggerCscId() const;

   /**
    * Lower and upper counts for the subdetector hierarchy
    */
   static int minEndcapId()  { return MIN_ENDCAP; }
   static int maxEndcapId()  { return MAX_ENDCAP; }
   static int minStationId() { return MIN_STATION; }
   static int maxStationId() { return MAX_STATION; }
   static int minRingId()    { return MIN_RING; }
   static int maxRingId()    { return MAX_RING; }
   static int minChamberId() { return MIN_CHAMBER; }
   static int maxChamberId() { return MAX_CHAMBER; }
   static int minLayerId()   { return MIN_LAYER; }
   static int maxLayerId()   { return MAX_LAYER; }

   /**
    * Returns the chamber name in the format
    * ME$sign$station/$ring/$chamber. Example: ME+1/1/9
    */
   static std::string chamberName(int endcap, int station, int ring, int chamber);
   std::string chamberName() const;

private:
 
  /**
   * Method for initialization within ctors.
   *
   */
  static uint32_t init( int iendcap, int istation, 
			int iring, int ichamber, int ilayer ) {
    
    if (istation == 1)
      iring = intToDetId(iring);

     return
         (ilayer   & MASK_LAYER)                      |
       ( (ichamber & MASK_CHAMBER) << START_CHAMBER ) |
       ( (iring    & MASK_RING)    << START_RING )    |
       ( (istation & MASK_STATION) << START_STATION ) | 
       ( (iendcap  & MASK_ENDCAP)  << START_ENDCAP ) ; }

  /**
   *
   * Methods for reordering CSCDetId for ME1 detectors.
   *
   * Internally the chambers are ordered (Station/Ring) as: ME1/a (1/1), ME1/b (1/2), ME1/2 (1/3), ME1/3 (1/4)
   * i.e. they are labelled within the DetId as if ME1a, ME1b, ME12, ME13 are rings 1, 2, 3, 4.
   * The offline software always considers rings 1, 2, 3, 4 as ME1b, ME12, ME13, ME1a so that at
   * least ME12 and ME13 have ring numbers which match in hardware and software!
   *
   */
  static int intToDetId(int iring) {
    // change iring = 1, 2, 3, 4 input to 2, 3, 4, 1 for use inside the DetId
    // i.e. ME1b, ME12, ME13, ME1a externally become stored internally in order ME1a, ME1b, ME12, ME13
    int i = (iring+1)%4;
    if (i == 0) i = 4;
    return i;
  }

  static int detIdToInt(int iring) {
    // reverse intToDetId: change 1, 2, 3, 4 inside the DetId to 4, 1, 2, 3 for external use
    // i.e. output ring # 1, 2, 3, 4 in ME1 means ME1b, ME12, ME13, ME1a as usual in the offline software.
    int i = (iring-1);
    if (i == 0) i = 4;
    return i;
  }
 
  // The following define the bit-packing implementation...

  // The maximum numbers of various parts
  enum eMaxNum{ MAX_ENDCAP=2, MAX_STATION=4, MAX_RING=4, MAX_CHAMBER=36, MAX_LAYER=6 };
  // We count from 1 
  enum eMinNum{ MIN_ENDCAP=1, MIN_STATION=1, MIN_RING=1, MIN_CHAMBER=1, MIN_LAYER=1 };

  // BITS_det is no. of binary bits required to label 'det' but allow 0 as a wild-card character
  // Keep as multiples of 3 so that number can be easily decodable from octal
  enum eNumBitDet{ BITS_ENDCAP=3, BITS_STATION=3,  BITS_RING=3, BITS_CHAMBER=6, BITS_LAYER=3 };

  // MASK_det is binary bits set to pick off the bits for 'det' (defined as octal)
  enum eMaskBitDet{ MASK_ENDCAP=07, MASK_STATION=07, MASK_RING=07, MASK_CHAMBER=077, MASK_LAYER=07 };

  // START_det is bit position (counting from zero) at which bits for 'det' start in 'rawId' word
  enum eStartBitDet{ START_CHAMBER=BITS_LAYER, START_RING=START_CHAMBER+BITS_CHAMBER,
          START_STATION=START_RING+BITS_RING, START_ENDCAP=START_STATION+BITS_STATION };
};

#endif



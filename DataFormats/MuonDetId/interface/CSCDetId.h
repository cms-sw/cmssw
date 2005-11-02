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
 * forth between a MuEndLayer 'rawId' and the %set of subdetector labels.
 *
 * \warning EVERY LABEL COUNTS FROM ONE NOT ZERO.
 *
 */

#include <iosfwd>
#include <DataFormats/DetId/interface/DetId.h>
#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>

class CSCDetId;

std::ostream& operator<<( std::ostream& os, const CSCDetId& id );

class CSCDetId : public DetId {

public:

  /// Default constructor; fills the common part in the base
  /// and leaves 0 in all other fields
  CSCDetId();

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is CSC, otherwise an exception is thrown.
  explicit CSCDetId(uint32_t id);


  /// Construct from fully qualified identifier.
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  CSCDetId( int iendcap, int istation, 
	    int iring, int ichamber, 
	    int ilayer );

  /** Copy ctor.
   */
  CSCDetId( const CSCDetId& id )
     : DetId(id.id_) { }  

   /**
    * Return Layer label.
    *
    */
    int layer() const {
     return (id_ & MASK_LAYER); } // counts from 1 not 0.

   /**
    * Return Chamber label.
    *
    */
    int chamber() const {
     return (  (id_>>START_CHAMBER) & MASK_CHAMBER ) + 1; }

   /**
    * Return Ring label.
    *
    */
    int ring() const {
     return (  (id_>>START_RING) & MASK_RING ) + 1; }

   /**
    * Return Station label.
    *
    */
    int station() const {
     return (  (id_>>START_STATION) & MASK_STATION ) + 1; }

   /**
    * Return Endcap label.
    *
    */
    int endcap() const {
     return (  (id_>>START_ENDCAP) & MASK_ENDCAP ) + 1; }


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

  // Tim dislikes the necessity of this ugly code - magic numbers included
  // Thanks a lot, CMSSW

   static int rawIdMaker( int iendcap, int istation, int iring, 
               int ichamber, int ilayer ) {
     return ((DetId::Muon&0xF)<<28)|((MuonSubdetId::CSC&0x7)<<25)|
               init(iendcap, istation, iring, ichamber, ilayer) ; }

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
     return (  (index>>START_CHAMBER) & MASK_CHAMBER ) + 1; }

   /**
    * Return Ring label for supplied CSCDetId index.
    *
    */
   static int ring( int index ) {
     return (  (index>>START_RING) & MASK_RING ) + 1; }

   /**
    * Return Station label for supplied CSCDetId index.
    *
    */
   static int station( int index ) {
     return (  (index>>START_STATION) & MASK_STATION ) + 1; }

   /**
    * Return Endcap label for supplied CSCDetId index.
    *
    */
   static int endcap( int index ) {
     return (  (index>>START_ENDCAP) & MASK_ENDCAP ) + 1; }

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


   /* Tim asks: Wouldn't it have been better to keep these as enums?
      Aren't nicely-packaged enum sets rather neat?
      Now we have a disconnected collection of values which are associated 
      only by the fact that each begins with 'min' or 'max'.               */

   /// lowest endcap id
   static const int minEndcapId=     1;
   /// highest endcap id
   static const int maxEndcapId=     2;
   static const int minStationId=    1;
   static const int maxStationId=    4;
   static const int minRingId=       1;
   static const int maxRingId=       4;
   static const int minChamberId=    1;
   static const int maxChamberId=   36;
   static const int minLayerId=      1; // Tim asks: Why was this set to 0?
   static const int maxLayerId=      6;
   
private:

  /**
   * Method for initialization within ctors.
   *
   */
  static uint32_t init( int iendcap, int istation, 
			int iring, int ichamber, int ilayer ) {
     return
                   ilayer + 
                ( (ichamber-1)<<START_CHAMBER ) + 
                   ( (iring-1)<<START_RING ) + 
                ( (istation-1)<<START_STATION ) + 
            	 ( (iendcap-1)<<START_ENDCAP ) ; }

 
 
  // The following define the bit-packing implementation...

  // This class is designed to handle the following maxima
  // There are 2 endcaps, which are the two z ends.
  //  enum eMaxNum{ MAX_RING=4, MAX_STATION=4, MAX_LAYER=6, MAX_CHAMBER=36};

  // BITS_det is no. of binary bits required to label 'det'
  enum eNumBitDet{ BITS_LAYER=3, BITS_CHAMBER=6, BITS_RING=2, 
                    BITS_STATION=2, BITS_ENDCAP=1};

  // MASK_det is binary bits set to pick off the bits for 'det'
  enum eMaskBitDet{ MASK_LAYER=07, MASK_CHAMBER=077, MASK_RING=03,
		     MASK_STATION=03, MASK_ENDCAP=01 };

  // START_det is bit position (counting from zero) at which bits for 'det' start in 'rawId' word
  enum eStartBitDet{ START_CHAMBER=BITS_LAYER, START_RING=START_CHAMBER+BITS_CHAMBER,
          START_STATION=START_RING+BITS_RING, START_ENDCAP=START_STATION+BITS_STATION };
};

#endif



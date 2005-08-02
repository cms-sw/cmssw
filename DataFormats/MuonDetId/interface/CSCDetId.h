#ifndef CSC_DETECTOR_ID_H
#define CSC_DETECTOR_ID_H

// This is CSCDetectorId.h

/** \class CSCDetectorId
 * Identifier class for hierarchy of Endcap Muon detector components.
 *
 * Allows access to hardware integer labels of the subcomponents
 * of the Muon Endcap detector system. The member functions  should rarely be
 * required by users since the subdetector objects already know their labels.
 *
 * However, the STATIC member functions can be used to translate back and
 * forth between a MuEndLayer 'index' and the %set of subdetector labels.
 *
 * \warning EVERY LABEL COUNTS FROM ONE NOT ZERO.
 *
 * \author Tim Cox
 * \author Rick Wilkinson
 *
 * Last mod: <BR>
 * 23-May-99 ptc Under development as a hierarchy...argh! This started as a
 *               simple %set of integers. <BR>
 * 27-Jul-99 ptc Remove MuEndLayer* - makes horrid package dependency
 *               and it's original purpose has vanished during development! <BR>
 * 01-Sep-99 ptc Add sector() and cscId() to MuEndChamberId <BR>
 * 15-Sep-99 ptc int->unsigned int <BR>
 * 21-Sep-99 ptc doxygenate. <BR>
 * 22-Sep-99 ptc Add lIndex to return unique integer index for each layer. <BR>
 * 06-Oct-99 ptc lIndex now const (Rick noticed) <BR>
 * 06-Nov-99 ptc Add mod hist to doxygen. <BR>
 * 15-Nov-99 ptc Update comments & code for sector() & cscId(). <BR>
 * 16-Nov-99 ptc Update sector() & cscId() to match new mu_end_chamber_index. <BR>
 * 21-Nov-99 ptc Bug-fix in cscId(). <BR>
 * 06-Feb-00 ptc Major redesign of MuEndDetectorId to use just one class level.
 *               This is a long-overdue but welcome simplification.
 *               Methods access subdetector component labels.
 *               Method 'index' with subdet label args returns unique int for use
 *               with digis and ntuples (replaces old 'lIndex'.) <BR>
 * 11-Feb-00 ptc Polish doxygen. <BR>
 * 2-Aug-05  sa  Renamed to CSCDetId.
 *               Added inheritance from DetId.
 *               Inserted into the new CMSSW structure <BR>
 *               Added data members to tell the bounds of the various 
 *               components (minStationId, etc)
 *   
 */

#include <iosfwd>
#include <DataFormats/DetId/interface/DetId.h>
#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>

class CSCDetId;

std::ostream& operator<<( std::ostream& os, const CSCDetId& id );

class CSCDetId:public cms::DetId {

  //@@ Unnecessary?
  //  friend ostream& operator<<( ostream& os, const CSCDetectorId& id ); 
  //@@ Iterator class needs access to Id class UNNEC IF PUBLIC
  //   friend class CSCLayerIterator; 

public:

  /** Explicit and default ctor.
   */
  CSCDetId( unsigned int iendcap=0, unsigned int istation=0, 
		    unsigned int iring=0, unsigned int ichamber=0, 
                     unsigned int ilayer=0 ) 
     : DetId(cms::DetId::Muon, MuonSubdetId::CSC )
    {id_ |= init(iendcap, istation, iring, ichamber, ilayer);  }

  /** Copy ctor.
   */
  CSCDetId( const CSCDetId& id )
     : DetId(id.id_) { }  

//  ~CSCDetId();                                        // Default dtor is ok
//  CSCDetId& operator=( const CSCDetId& rhs );  // Default assignment op is ok

  bool operator == ( const CSCDetId& ) const;
  bool operator != ( const CSCDetId& ) const;
  bool operator <  ( const CSCDetId& ) const;

  /** 
   * Unique integer index for this CSCDetId.
   *
   * \warning The returned integers are not necessarily consecutive, i.e.
   * there are gaps. 
   *
   */
   unsigned int index( void ) const {
     return id_; // This member already encodes the info.
   }          

   /**
    * Return Layer label.
    *
    */
   unsigned int layer( void ) const {
     //     return (id_ & MASK_LAYER) + 1; }
     return (id_ & MASK_LAYER); } // index counts from 1 not 0.

   /**
    * Return Chamber label.
    *
    */
   unsigned int chamber( void ) const {
     return (  (id_>>START_CHAMBER) & MASK_CHAMBER ) + 1; }

   /**
    * Return Ring label.
    *
    */
   unsigned int ring( void ) const {
     return (  (id_>>START_RING) & MASK_RING ) + 1; }

   /**
    * Return Station label.
    *
    */
   unsigned int station( void ) const {
     return (  (id_>>START_STATION) & MASK_STATION ) + 1; }

   /**
    * Return Endcap label.
    *
    */
   unsigned int endcap( void ) const {
     return (  (id_>>START_ENDCAP) & MASK_ENDCAP ) + 1; }


   // static methods
   // Used when we need information about subdetector labels.

  /** 
   * Unique integer index for each Layer.
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
   static unsigned int index( unsigned int iendcap, unsigned int istation, unsigned int iring, 
               unsigned int ichamber, unsigned int ilayer ) {
     return init(iendcap, istation, iring, ichamber, ilayer) ; }

   /**
    * Return Layer label for supplied CSCDetId index.
    *
    */
   static unsigned int layer( unsigned int index ) {
     //     return (index & MASK_LAYER) + 1; }
     return (index & MASK_LAYER); } // index counts from 1 not 0

   /**
    * Return Chamber label for supplied CSCDetId index.
    *
    */
   static unsigned int chamber( unsigned int index ) {
     return (  (index>>START_CHAMBER) & MASK_CHAMBER ) + 1; }

   /**
    * Return Ring label for supplied CSCDetId index.
    *
    */
   static unsigned int ring( unsigned int index ) {
     return (  (index>>START_RING) & MASK_RING ) + 1; }

   /**
    * Return Station label for supplied CSCDetId index.
    *
    */
   static unsigned int station( unsigned int index ) {
     return (  (index>>START_STATION) & MASK_STATION ) + 1; }

   /**
    * Return Endcap label for supplied CSCDetId index.
    *
    */
   static unsigned int endcap( unsigned int index ) {
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
   unsigned int sector() const;

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
   unsigned int cscId() const;

   /// lowest endcap id
   static const unsigned int minEndcapId=     1;
   /// highest endcap id
   static const unsigned int maxEndcapId=     2;
   static const unsigned int minStationId=    1;
   static const unsigned int maxStationId=    4;
   static const unsigned int minRingId=       1;
   static const unsigned int maxRingId=       4;
   static const unsigned int minChamberId=    1;
   static const unsigned int maxChamberId=   36;
   static const unsigned int minLayerId=      1;
   static const unsigned int maxLayerId=      6;
   
private:

  /**
   * Method for initialization within ctors.
   *
   */
  static unsigned int init( unsigned int iendcap, unsigned int istation, 
          unsigned int iring, unsigned int ichamber, unsigned int ilayer ) {
     return 
       //                    (ilayer-1) + // Oops! -1 means I start at 0. Wrong!
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

  // START_det is bit position (counting from zero) at which bits for 'det' start in 'index' word
  enum eStartBitDet{ START_CHAMBER=BITS_LAYER, START_RING=START_CHAMBER+BITS_CHAMBER,
          START_STATION=START_RING+BITS_RING, START_ENDCAP=START_STATION+BITS_STATION };
};

#endif



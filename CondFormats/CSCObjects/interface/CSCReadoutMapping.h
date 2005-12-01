#ifndef CondFormats_CSCReadoutMapping_h
#define CondFormats_CSCReadoutMapping_h

/** 
 * \class CSCReadoutMapping
 * \author Tim Cox
 * Abstract class to define mapping between CSC readout hardware ids and other labels.
 *
 * Defines the ids and labels in the mapping and supplies tramslation interface.
 * How the mapping is filled, and where from, depends on a concrete class.
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <map>

class CSCReadoutMapping {
 public:

  /// Default constructor
  CSCReadoutMapping();

  /// Destructor
  virtual ~CSCReadoutMapping();

  /**
   * Instead of a set of vectors of int use one vector of a set of ints
   */  
  struct CSCLabel{

    CSCLabel( int endcap, int station, int ring, int chamber,  
	      int vmecrate, int dmb, int tmb, int tsector, int cscid )
      : endcap_( endcap ), station_( station ), ring_( ring ), chamber_( chamber ),
         vmecrate_( vmecrate ), dmb_( dmb ), tmb_( tmb ), 
	 tsector_( tsector ), cscid_( cscid ) {}
    ~CSCLabel(){}

    int endcap_;
    int station_;
    int ring_;
    int chamber_;
    int vmecrate_;
    int dmb_;
    int tmb_;
    int tsector_;
    int cscid_;
  };


  /**
   * Return CSCDetId for layer corresponding to readout ids vme, tmb, and dmb for given endcap
   * and layer no. 1-6, or for chamber if no layer no. supplied.
   * Args: endcap = 1 (+z), 2 (-z), station, vme crate number, dmb slot number, tmb slot number, layer#
   */
  // layer at end so it can have default arg
  CSCDetId detId( int endcap, int station, int vmecrate, int dmb, int tmb, int layer = 0 );

   /** 
    * Return chamber label corresponding to readout ids vme, tmb and dmb for given endcap
    *  endcap = 1 (+z), 2 (-z), vme crate number, tmb slot number, dmb slot number
    */
  int chamber( int endcap, int station, int vmecrate, int dmb, int tmb );

    //@@ FIXME further interface as required to handle any mapping one likes,
    //@@ as long as it's feasible from the available labels.

   /** 
    * Fill mapping store
    */
    virtual void fill( void ) = 0;

   /**
    * Add a CSCLabel element to mapping     
    */
    //    void addRecord( CSCLabel label );    

   /**
    * Add one record of info to mapping
    */
    void addRecord( int endcap, int station, int ring, int chamber, 
		    int vmecrate, int dmb, int tmb, int tsector, int cscid ); 

    /**
     * Set debug printout flag
     */
    void setDebugV( bool dbg ) { debugV_ = dbg; }

    /**
     * Status of debug printout flag
     */
    bool debugV( void ) { return debugV_; }

 private: 

    /**
     * Build a unique integer out of the readout electronics labels.
     *
     * In general this must depend on endcap and station, as well as
     * vme crate number and dmb slot number. In principle perhaps tmb slot
     * number might not be neighbour of dmb?
     * But for slice test (Nov-2005) only relevant labels are vme and dmb.
     */
    int hwId( int endcap, int station, int vme, int dmb, int tmb );

    /**
     * Build a unique integer out of chamber labels.
     *
     * We'll probably use rawId of CSCDetId... You know it makes sense!
     */
    int swId( int endcap, int station, int ring, int chamber);

    const std::string myName_;
    bool debugV_;
    std::vector< CSCLabel > mapping_;
    std::map< int, int > hw2sw_;

};

#endif

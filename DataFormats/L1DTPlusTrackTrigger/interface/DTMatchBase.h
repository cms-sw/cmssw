#ifndef DTMatchBase_h
#define DTMatchBase_h

/*! \class DTMatchBase
 *  \author Ignazio Lazzizzera
 *  \author Nicola Pozzobon
 *  \brief DT local triggers matched together, base class
 *         Used to store detector-related information and
 *         matched tracker objects
 *  \date 2010, Apr 10
 */

#include "DataFormats/MuonDetId/interface/DTBtiId.h"
#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatchBasePtMethods.h"

#include <map>
#include <vector>

unsigned int const numberOfTriggerLayers = 6;

/// Class implementation
class DTMatchBase : public DTMatchBasePtMethods
{
  public :

    /// Method with the only purpose of making this class abstract
//    virtual void toMakeThisAbstract() = 0;

    /// Trivial constructor
    DTMatchBase();

    /// Non-trivial constructors
    DTMatchBase( int aDTWheel, int aDTStation, int aDTSector,
                 int aDTBX, int aDTCode, int aTSPhi, int aTSPhiB, int aTSTheta,
                 bool aFlagBXOK );

    DTMatchBase( int aDTWheel, int aDTStation, int aDTSector,
                 int aDTBX, int aDTCode, int aTSPhi, int aTSPhiB, int aTSTheta,
                 GlobalPoint aDTPosition, GlobalVector aDTDirection,
                 bool aFlagBXOK );

    /// Copy constructor
    DTMatchBase( const DTMatchBase& aDTMB );

    /// Assignment operator
    DTMatchBase& operator = ( const DTMatchBase& aDTMB );

    /// Destructor
    virtual ~DTMatchBase(){};

    /// Return functions for muon trigger information
    inline int getDTWheel() const { return theDTWheel; }
    inline int getDTStation() const { return theDTStation; }
    inline int getDTSector() const { return theDTSector; }
    inline int getDTBX() const { return theDTBX; }
    inline bool getFlagBXOK() const { return theFlagBXOK; }
    inline int getDTCode() const { return theDTCode; }
    inline int getDTTSPhi() const { return theTSPhi; }
    inline int getDTTSPhiB() const { return theTSPhiB; }
    inline int getDTTSTheta() const { return theTSTheta; }
    inline DTBtiId getInnerBtiId() const { return theInnerBti; }
    inline DTBtiId getOuterBtiId() const { return theOuterBti; }
    inline DTBtiId getMatchedBtiId() const { return theMatchedBti; }

    void setInnerBtiId( DTBtiId anId )
    {
      theInnerBti = anId;
      return;
    }

    void setOuterBtiId( DTBtiId anId )
    {
      theOuterBti = anId;
      return;
    }

    void setMatchedBtiId( DTBtiId anId )
    {
      theMatchedBti = anId;
      return;
    }

    /// Return functions for muon trigger position and direction
    inline GlobalPoint getDTPosition() const { return theDTPosition; }
    inline GlobalVector getDTDirection() const { return theDTDirection; }

    /// Return function for other data members that are internally calculated
    inline float getAlphaDT() const { return theAlphaDT; }

    /// Return functions for virtual Tracker layer
//    inline double getInnerCoilR() const { return theInnerCoilR; }
//    inline double getOuterCoilR() const { return theOuterCoilR; }
//    inline double getCoilRTilde() const { return theCoilRTilde; }

    /**** STUBS ****/
    /// Store the stub
    inline void addMatchedStubRef( edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > aStub, GlobalPoint aStubPos )
    {
      StackedTrackerDetId thisId( aStub->getDetId() );
      theMatchedStubRef.insert( std::make_pair( thisId.iLayer(), aStub ) );
      theMatchedStubPos.insert( std::make_pair( thisId.iLayer(), aStubPos ) );
      return;
    }

    /// Get all the stubs matched to the DT object
    inline std::map< unsigned int, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > getMatchedStubRefs() const
    {
      return theMatchedStubRef;
    }

    /// Get the positions of all the stubs matched to the DT object
    inline std::map< unsigned int, GlobalPoint > getMatchedStubPositions() const
    {
      return theMatchedStubPos;
    }

    /**** TRACKS ****/
    /// Store a track matched in direction (in the window)
    inline void addInWindowTrackPtr( edm::Ptr< TTTrack< Ref_PixelDigi_ > > aTrack )
    {
      theTrackPtrInWindow.push_back( aTrack );
      return;
    }

    /// Get all the track matched (in the window) to the DT object
    inline std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > getInWindowTrackPtrs() const
    {
      return theTrackPtrInWindow;
    }

    /// Store the track matched (Pt)
    inline void setPtMatchedTrackPtr( edm::Ptr< TTTrack< Ref_PixelDigi_ > > aTrack )
    {
      theMatchedTrackPtr = aTrack;
      return;
    }

    /// Get the track matched (Pt) to the DT object
    inline edm::Ptr< TTTrack< Ref_PixelDigi_ > > getPtMatchedTrackPtr() const
    {
      return theMatchedTrackPtr;
    }

    /// Method to create all the DTMatchPt objects that
    /// are stored in the DTMatchBasePtMethods (this class is
    /// inheriting from that class)
    /// For each method, it creates the object, calculates the Pt,
    /// and stores it using DTMatchBasePtMethods::addPtMethod( ... );
    void setPtMethods( float station2Correction, bool thirdMethodAccurate,
                       float aMinRInvB, float aMaxRInvB );


    /// Methods that convert from integer values to
    /// global CMS floating point numbers
    inline float getGlobalTSPhi() const
    {
      float phiCMS = static_cast< float >( theTSPhi )/4096. + ( theDTSector - 1 ) * M_PI / 6.;

      /// Renormalize phi
      if ( phiCMS <= 0. )
      {
        phiCMS += 2. * M_PI;
      }
      if( phiCMS > 2. * M_PI )
      {
        phiCMS -= 2. * M_PI;
      }

      return phiCMS;
    }

    inline float getGlobalTSPhiB() const
    {
      return static_cast< float >( theTSPhiB ) / 512.; /// 9 bits
    }

  private :

    /// Data members

    /// The original DT trigger information
    int theDTWheel, theDTStation, theDTSector, theDTBX, theDTCode;
    int theTSPhi, theTSTheta, theTSPhiB;
    bool theFlagBXOK;

    DTBtiId theInnerBti, theOuterBti; /// As all the matches start from a TSPhi
                                      /// here we store the ID's of the two BTI's that fired the DTChambPhSegm
    DTBtiId theMatchedBti; /// There are 3 possible matches: to a BTI, to a TSTheta (one BTI from the TSTheta)
                           /// and from nothing (in such a case, it is left to its default value = 0x0

    /// The original DT trigger position and direction
    GlobalPoint theDTPosition;
    GlobalVector theDTDirection;

    /// Internally computed data members
    float theAlphaDT;

    /// Virtual Tracker layer at solenoid
    static const double theInnerCoilR, theOuterCoilR, theCoilRTilde; 

    /// Matching TT objects
    std::map< unsigned int, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > theMatchedStubRef;
    std::map< unsigned int, GlobalPoint > theMatchedStubPos;
    std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > theTrackPtrInWindow;
    edm::Ptr< TTTrack< Ref_PixelDigi_ > > theMatchedTrackPtr;
};

#endif


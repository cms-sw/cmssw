#ifndef DataFormats_CSCRecHit2D_H
#define DataFormats_CSCRecHit2D_H

/**
 * \class CSCRecHit2D  
 * Describes a 2-dim reconstructed hit in one layer of an Endcap Muon CSC.
 *
 * \author Tim Cox et al.
 *
 */
#include "DataFormats/Common/interface/RangeMap.h"
#include <DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <map>
#include <iosfwd>

class CSCRecHit2D GCC11_FINAL : public RecHit2DLocalPos {

public:

  typedef std::vector<int> ChannelContainer;
  typedef edm::RangeMap<int, std::vector<float> > ADCContainer;
  
  enum SharedInputType {all = TrackingRecHit::all, some = TrackingRecHit::some, allWires, someWires, allStrips, someStrips};

  static const unsigned int MAXSTRIPS=3;
  static const unsigned int MAXTIMEBINS=4;
  static const unsigned int N_ADC=MAXSTRIPS*MAXTIMEBINS;
  CSCRecHit2D();

  CSCRecHit2D( const CSCDetId& id, 
               const LocalPoint& pos, const LocalError& err, 
	       const ChannelContainer& channels,
	       const ADCContainer& adcs,
	       const ChannelContainer& wgroups,
               float tpeak,
               float posInStrip,
               float errInStrip,
	       int quality,
               short int badStrip=0, short int badWireGroup=0,
               int scaledWireTime=0,
	       float energyDeposit=-995.); 
	
  ~CSCRecHit2D();


  /// RecHit2DLocalPos base class interface
  CSCRecHit2D* clone() const { return new CSCRecHit2D( *this ); }
  LocalPoint localPosition() const { return theLocalPosition; }
  LocalError localPositionError() const { return theLocalError; }
  CSCDetId cscDetId() const { return geographicalId(); }

  /// Extracting strip channel numbers comprising the rechit - low
  int channels(unsigned int i) const { return theStrips_[i]; }
  unsigned int nStrips() const {return nStrips_;}

  /// Extract the L1A phase bits from the StripChannelContainer - high
  int channelsl1a(unsigned int i) const { return theL1APhaseBits_[i]; } /// L1A

  /// Container of wire groups comprising the rechit
  short int hitWire() const { return hitWire_;}
  short int wgroupsBX() const {return theWGroupsBX_;}

  unsigned int nWireGroups() const {return nWireGroups_;}

  /// Map of strip ADCs for strips comprising the rechit
  float adcs(unsigned int strip, unsigned int timebin) const { return theADCs_[strip*MAXTIMEBINS+timebin]; }

  unsigned int nTimeBins() const {return nTimeBins_;}

  /// Fitted peaking time
  float tpeak() const { return theTpeak; }

  /// The estimated position within the strip
  float positionWithinStrip() const { return thePositionWithinStrip; };

  /// The uncertainty of the estimated position within the strip
  float errorWithinStrip() const { return theErrorWithinStrip;} ;

  /// quality flag of the reconstruction
  int quality() const { return theQuality;}

  /// flags for involvement of 'bad' channels
  short int badStrip() const { return theBadStrip; }
  short int badWireGroup() const { return theBadWireGroup; }
  
  // Calculated wire time in ns
  float wireTime() const { return (float)theScaledWireTime/100.; }

  /// Energy deposited in the layer.  Note:  this value is dE.  In order to 
  /// get the dE/dX, you will need to divide by the path length.
  /// Specific failure values...
  /// If the user has chosen not to use the gas gain correction --->  -998.
  /// If the gas gain correction from the database is a bad value ->  -997.
  /// If it is an edge strip -------------------------------------->  -996.
  /// If gas-gain is OK, but the ADC vector is the wrong size  ---->  -999.
  /// If the user has created the Rechit without the energy deposit>  -995.
  /// If the user has created the Rechit with no arguments -------->  -994.
  float energyDepositedInLayer() const { return theEnergyDeposit; }

  /// Returns true if the two TrackingRecHits are using the same input information, false otherwise.  In this case, looks at the geographical ID and channel numbers for strips and wires.
  virtual bool sharesInput(const TrackingRecHit *other, TrackingRecHit::SharedInputType what) const;
  
  /// Returns true if the two TrackingRecHits are using the same input information, false otherwise.  In this case, looks at the geographical ID and channel numbers for strips and wires.
  bool sharesInput(const TrackingRecHit *other, CSCRecHit2D::SharedInputType what) const;

  /// Returns true if the two CSCRecHits are using the same input information, false otherwise.  In this case, looks at the geographical ID and channel numbers for strips and wires.
  bool sharesInput(const  CSCRecHit2D *otherRecHit, CSCRecHit2D::SharedInputType what) const;
   
   /// Print the content of the RecHit2D including L1A (for debugging)
   void print() const;	

private:
	
  float theTpeak;
  float thePositionWithinStrip; 
  float theErrorWithinStrip;
  float theEnergyDeposit;
  int theQuality;
  int theScaledWireTime;
  short int hitWire_;
  short int theWGroupsBX_;
  short int theBadStrip;
  short int theBadWireGroup;

  unsigned char nStrips_, nWireGroups_, nTimeBins_;

  unsigned char theL1APhaseBits_[MAXSTRIPS];
  unsigned char theStrips_[MAXSTRIPS];
  float theADCs_[N_ADC];

  LocalPoint theLocalPosition;
  LocalError theLocalError;

};

/// Output operator for CSCRecHit2D
std::ostream& operator<<(std::ostream& os, const CSCRecHit2D& rh);

#endif


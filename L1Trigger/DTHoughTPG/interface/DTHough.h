#ifndef DTHT_DTHough_H
#define DTHT_DTHough_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "L1Trigger/DTHoughTPG/interface/Constants.h"

typedef edm::Ref< DTDigiCollection, DTDigi > RefDTDigi_t;

template< typename T >
class DTHough
{
  public :

    DTHough();
    ~DTHough();

    std::vector< T > GetHits() const { return theHits; }
    void             SetHits( std::vector< T > someHits ) { theHits = someHits; }
    std::vector< uint8_t > GetHitLayers() const { return theHitLayers; }
    void                   SetHitLayers( std::vector< uint8_t > someHitLayers ) { theHitLayers = someHitLayers; }

    DTChamberId  GetDTChamberId() const { return theChamberId; }
    void         SetDTChamberId( DTChamberId aChamberId ) { theChamberId = aChamberId; }
    unsigned int GetFiredSuperLayers() const { return theFiredSuperLayers; }
    void         SetFiredSuperLayers( unsigned int aFiredSuperLayers ) { theFiredSuperLayers = aFiredSuperLayers; }

    unsigned int GetMacroCellCode() const { return theMacroCellCode; }
    void         SetMacroCellCode( unsigned int aMacroCellCode ) { theMacroCellCode = aMacroCellCode; }
    double GetMCellCentralCoordinate() const { return theMCellCentralCoord; }
    void   SetMCellCentralCoordinate( double aMCellCentralCoord ) { theMCellCentralCoord = aMCellCentralCoord; }

    uint32_t     GetWireWord() const { return theWireWord; }
    void         SetWireWord( uint32_t aWireWord ) { theWireWord = aWireWord; }
    void         ModWireWord( uint32_t aWireWord ) { theWireWord = ( theWireWord << 16 ) | aWireWord; }

    double GetTrigTimeDiff() const { return theTrigTimeDiff; } /// Looks like it is already in ns.
    void   SetTrigTimeDiff( double aTrigTimeDiff ) { theTrigTimeDiff = aTrigTimeDiff; }
    int32_t GetBXTime() const { return theBXTime; } /// 1 BX is 32 units long, needs conversion in ns.
    void    SetBXTime( int32_t aBXTime ) { theBXTime = aBXTime; }

    int32_t Get2TanPhi128() const { return the2TanPhi128; }
    void    Set2TanPhi128( int32_t a2TanPhi128 ) { the2TanPhi128 = ( a2TanPhi128 != DEF_TanPhi ) ? a2TanPhi128 : 0x7FFFFFFF; }
    int32_t GetXMCell( int aSuperLayer ) const { assert( aSuperLayer > 0 && aSuperLayer < 4 ); return theXMCellList.at( aSuperLayer - 1 ); }
    void    SetXMCell( int32_t aXMCell, int aSuperLayer ) { assert( aSuperLayer > 0 && aSuperLayer < 4 ); theXMCellList[ aSuperLayer - 1 ] = ( aXMCell != DEF_MCellPos ) ? aXMCell : 0x7FFFFFFF; }

    /// End-user information
    double GetTrigTanPhi() const;
    double GetTrigX0() const;
    double GetUnprojectedTrigX0() const;
    double GetGlobalPhi() const;// { return theGlobalPhi; }
    void   SetGlobalPhi( double aGlobalPhi ) { theGlobalPhi = aGlobalPhi; }
    double GetSectorPhi() const;// { return theSectorPhi; }
    void   SetSectorPhi( double aSectorPhi ) { theSectorPhi = aSectorPhi; }
    double GetGlobalTheta() const { return theGlobalTheta; }
    void   SetGlobalTheta( double aGlobalTheta ) { theGlobalTheta = aGlobalTheta; }
    double GetBendingPhi() const;

    double GetGlobalPhiSL() const;// { return theGlobalPhi; }
    void   SetGlobalPhiSL( double aGlobalPhi ) { theGlobalPhiSL = aGlobalPhi; }
    double GetSectorPhiSL() const;// { return theSectorPhi; }
    void   SetSectorPhiSL( double aSectorPhi ) { theSectorPhiSL = aSectorPhi; }
    double GetBendingPhiSL() const;

    unsigned int GetQuality() const { return theQuality; }
    void         SetQuality( unsigned int aQuality ) { theQuality = aQuality; }

    std::string print() const;

  private :

    std::vector< T > theHits;
    std::vector< uint8_t > theHitLayers;
    DTChamberId theChamberId;

    unsigned int theFiredSuperLayers;
    double theTrigTimeDiff;
    unsigned int theMacroCellCode; /// Encoding is macro-cell if only one, (SL1 + 128 * SL2 + 128**2 * SL3) if two

    uint32_t theWireWord;
    int32_t theBXTime;
    int32_t the2TanPhi128; /// As it comes from the m-bitset
    std::vector< int32_t > theXMCellList;

    double theMCellCentralCoord;

    double theTrigTanPhi;
    double theTrigX0;
    unsigned int theQuality;
    double theGlobalPhi;
    double theSectorPhi; /// +4*M_PI for positive wheels
    double theGlobalTheta;

    double theGlobalPhiSL;
    double theSectorPhiSL;

}; /// Close class

template< typename T >
DTHough< T >::DTHough()
{
  /// Initialize data members
  theHits.clear();
  theHitLayers.clear();
  theChamberId = DTChamberId();
  theFiredSuperLayers = 999;
  theTrigTimeDiff = 0.0;
  theMacroCellCode = 0;
  theWireWord = 0;
  theBXTime = 0x7FFFFFFF;
  the2TanPhi128 = 0x7FFFFFFF;
  theXMCellList.assign(3,0x7FFFFFFF);
  theMCellCentralCoord = 999.9;
  theTrigTanPhi = 999.9;
  theTrigX0 = -999.9;
  theQuality = qDummy;
  theGlobalPhi = -999.9;
  theSectorPhi = -999.9;
  theGlobalPhiSL = -999.9;
  theSectorPhiSL = -999.9;
  theGlobalTheta = -999.9;
}

template< typename T >
DTHough< T >::~DTHough()
{
  theHits.clear();
  theHitLayers.clear();
}

/// End-user information
template< typename T >
double DTHough< T >::GetTrigTanPhi() const
{
  if ( theXMCellList[0] != 0x7FFFFFFF &&
       theXMCellList[2] != 0x7FFFFFFF )
  {
    return static_cast< double >( theXMCellList[0] - theXMCellList[2] ) * defUnitX / defDTSuperLayerDistance;
  }
  else
  {
    if ( the2TanPhi128 != 0x7FFFFFFF )
      return 0.5 * ( static_cast< double >( the2TanPhi128 ) / 128. );
    else
      return 999.9;
  }
}

template< typename T >
double DTHough< T >::GetTrigX0() const
{
  if ( theXMCellList[0] != 0x7FFFFFFF &&
       theXMCellList[2] != 0x7FFFFFFF )
    return 0.5 * static_cast< double >( theXMCellList[0] + theXMCellList[2] ) * defUnitX + theMCellCentralCoord;
  else
  {
    if ( theXMCellList[0] != 0x7FFFFFFF && the2TanPhi128 != 0x7FFFFFFF )
      return static_cast< double >( theXMCellList[0] ) * defUnitX + theMCellCentralCoord
        - 0.25 * defDTSuperLayerDistance * ( static_cast< double >( the2TanPhi128 ) / 128. );

    else if ( theXMCellList[2] != 0x7FFFFFFF && the2TanPhi128 != 0x7FFFFFFF )
      return static_cast< double >( theXMCellList[2] ) * defUnitX + theMCellCentralCoord
        + 0.25 * defDTSuperLayerDistance * ( static_cast< double >( the2TanPhi128 ) / 128. );

    else if ( theXMCellList[1] != 0x7FFFFFFF && the2TanPhi128 != 0x7FFFFFFF )
      return static_cast< double >( theXMCellList[1] ) * defUnitX + theMCellCentralCoord;

    else
      return 999.9;
  }
}

template< typename T >
double DTHough< T >::GetUnprojectedTrigX0() const
{
  if ( theXMCellList[0] != 0x7FFFFFFF &&
       theXMCellList[2] != 0x7FFFFFFF )
    return 0.5 * static_cast< double >( theXMCellList[0] + theXMCellList[2] ) * defUnitX + theMCellCentralCoord;
  else
  {
    if ( theXMCellList[0] != 0x7FFFFFFF && the2TanPhi128 != 0x7FFFFFFF )
      return static_cast< double >( theXMCellList[0] ) * defUnitX + theMCellCentralCoord;

    else if ( theXMCellList[2] != 0x7FFFFFFF && the2TanPhi128 != 0x7FFFFFFF )
      return static_cast< double >( theXMCellList[2] ) * defUnitX + theMCellCentralCoord;

    else if ( theXMCellList[1] != 0x7FFFFFFF && the2TanPhi128 != 0x7FFFFFFF )
      return static_cast< double >( theXMCellList[1] ) * defUnitX + theMCellCentralCoord;

    else
      return 999.9;
  }
}

template< typename T >
std::ostream& operator << ( std::ostream& os, const DTHough< T >& aDTHough ) { return ( os << aDTHough.print() ); }

template< typename T >
double DTHough< T >::GetGlobalPhi() const
{
  if ( theGlobalPhi < 0 )
    return theGlobalPhi + 2 * M_PI;
  return theGlobalPhi;
}

template< typename T >
double DTHough< T >::GetSectorPhi() const
{
  if ( theSectorPhi > 2.*M_PI )
    return theSectorPhi - 4.*M_PI; /// To switch between positive and negative wheels
  return theSectorPhi;
}

template< typename T >
double DTHough< T >::GetBendingPhi() const
{
  if ( this->GetTrigTanPhi() != 999.9 )
  {
    if ( theSectorPhi > 2.*M_PI )
    {
      return atan( this->GetTrigTanPhi() ) - this->GetSectorPhi();
    }
    else
      return - atan( this->GetTrigTanPhi() ) - this->GetSectorPhi();
  }
  return 999.9;
}

template< typename T >
double DTHough< T >::GetGlobalPhiSL() const
{
  if ( theGlobalPhiSL < 0 )
    return theGlobalPhiSL + 2 * M_PI;
  return theGlobalPhiSL;
}

template< typename T >
double DTHough< T >::GetSectorPhiSL() const
{
  if ( theSectorPhiSL > 2.*M_PI )
    return theSectorPhiSL - 4.*M_PI; /// To switch between positive and negative wheels
  return theSectorPhiSL;
}

template< typename T >
double DTHough< T >::GetBendingPhiSL() const
{
  if ( this->GetTrigTanPhi() != 999.9 )
  {
    if ( theSectorPhiSL > 2.*M_PI )
    {
      return atan( this->GetTrigTanPhi() ) - this->GetSectorPhiSL();
    }
    else
      return - atan( this->GetTrigTanPhi() ) - this->GetSectorPhiSL();
  }
  return 999.9;
}

/// Print info
template< typename T >
std::string DTHough< T >::print() const
{
  std::stringstream output;
  output << "\t" << "===============\n";
  output << "DTHough:\n";
  output << "\t" << "DetId:\t" << theChamberId.rawId() << "\n";
  output << "\t" << "fired SL:\t" << theFiredSuperLayers << "\n";
  output << "\t" << "macro-cell:\t" << theMacroCellCode % 128 << "\n";
  output << "\t" << "           \t" << ( theMacroCellCode % 16384 ) / 128 << "\n";
  output << "\t" << "           \t" << theMacroCellCode / 16384 << "\n";
  output << "\t" << "quality:\t" << theQuality << "\n";
  output << "\t" << "num hits:\t" << theHits.size() << "\n";
  output << "\t" << "time diff:\t" << theTrigTimeDiff << "\n";
  if ( theTrigTimeDiff == 0.0 )
    output << "\t\t" << "IN TIME!!!!!!!!!!!!!\n";
  output << "\t" << "2tan(phi) from HT:\t" << the2TanPhi128 << "\n";
  output << "\t" << "x0 from HT 1:\t\t" << this->GetXMCell(1) << "\n";
  output << "\t" << "x0 from HT 3:\t\t" << this->GetXMCell(3) << "\n";
  output << "\t" << "tan phi calc:\t\t" << this->GetTrigTanPhi() << "\n";
  output << "\t" << "x0 at chamb 0:\t\t" << this->GetTrigX0() << "\n";
  output << "\t" << "--------------\n";
  output << "\t" << "global phi:\t\t" << this->GetGlobalPhi() << "\n";
  output << "\t" << "sector phi:\t\t" << this->GetSectorPhi() << "\n";
  output << "\t" << "global theta:\t\t" << this->GetGlobalTheta() << "\n";
  output << "\t" << "bending angle:\t\t" << this->GetBendingPhi() << "\n";
  output << "\n";
  return output.str();
}

#endif

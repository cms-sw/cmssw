// -*- C++ -*-
//
// Package:     L1Geometry
// Class  :     L1CaloGeometry
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Werner Sun
//         Created:  Mon Oct 23 21:52:36 EDT 2006
// $Id: L1CaloGeometry.cc,v 1.5 2009/10/30 03:33:28 wsun Exp $
//

// system include files
#include <cmath>

// user include files
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

// double L1CaloGeometry::m_gctEmJetPhiOffset =
//    -M_PI / L1CaloGeometry::kNumberGctEmJetPhiBins ;
// double L1CaloGeometry::m_gctEtSumPhiOffset = 0. ;

// double L1CaloGeometry::m_gctEmJetPhiBinWidth =
//    2. * M_PI / L1CaloGeometry::kNumberGctEmJetPhiBins ;
// double L1CaloGeometry::m_gctEtSumPhiBinWidth =
//    2. * M_PI / L1CaloGeometry::kNumberGctEtSumPhiBins ;

// double L1CaloGeometry::m_gctEtaBinBoundaries[
//    kNumberGctCentralEtaBinsPerHalf + kNumberGctForwardEtaBinsPerHalf + 1 ] = {
//       0.0000,
//       0.3480,
//       0.6950,
//       1.0440,
//       1.3920,
//       1.7400,
//       2.1720,
//       3.0000,
//       3.5000,
//       4.0000,
//       4.5000,
//       5.0000 } ;

//
// constructors and destructor
//
L1CaloGeometry::L1CaloGeometry()
  : m_version( kOrig ), // if version is not in CondDB, set it to kOrig
    m_numberGctEmJetPhiBins( 0 ),
    m_numberGctEtSumPhiBins( 0 ),
    m_numberGctHtSumPhiBins( 0 ),
    m_numberGctCentralEtaBinsPerHalf( 0 ),
    m_numberGctForwardEtaBinsPerHalf( 0 ),
    m_etaSignBitOffset( 0 ),
    m_gctEtaBinBoundaries(),
    m_etaBinsPerHalf( 0 ),
    m_gctEmJetPhiBinWidth( 0. ),
    m_gctEtSumPhiBinWidth( 0. ),
    m_gctHtSumPhiBinWidth( 0. ),
    m_gctEmJetPhiOffset( 0. ),
    m_gctEtSumPhiOffset( 0. ),
    m_gctHtSumPhiOffset( 0. )
{
}

L1CaloGeometry::L1CaloGeometry( unsigned int numberGctEmJetPhiBins,
				double gctEmJetPhiBinOffset,
				unsigned int numberGctEtSumPhiBins,
				double gctEtSumPhiBinOffset,
				unsigned int numberGctHtSumPhiBins,
				double gctHtSumPhiBinOffset,
				unsigned int numberGctCentralEtaBinsPerHalf,
				unsigned int numberGctForwardEtaBinsPerHalf,
				unsigned int etaSignBitOffset,
				const std::vector<double>& gctEtaBinBoundaries)
  : m_version( kAddedMHTPhi ),
    m_numberGctEmJetPhiBins( numberGctEmJetPhiBins ),
    m_numberGctEtSumPhiBins( numberGctEtSumPhiBins ),
    m_numberGctHtSumPhiBins( numberGctHtSumPhiBins ),
    m_numberGctCentralEtaBinsPerHalf( numberGctCentralEtaBinsPerHalf ),
    m_numberGctForwardEtaBinsPerHalf( numberGctForwardEtaBinsPerHalf ),
    m_etaSignBitOffset( etaSignBitOffset ),
    m_gctEtaBinBoundaries( gctEtaBinBoundaries )
{
  m_etaBinsPerHalf =
    m_numberGctCentralEtaBinsPerHalf + m_numberGctForwardEtaBinsPerHalf ;

  m_gctEmJetPhiBinWidth = 2. * M_PI / m_numberGctEmJetPhiBins ;
  m_gctEtSumPhiBinWidth = 2. * M_PI / m_numberGctEtSumPhiBins ;
  m_gctHtSumPhiBinWidth = 2. * M_PI / m_numberGctHtSumPhiBins ;

  m_gctEmJetPhiOffset = gctEmJetPhiBinOffset * m_gctEmJetPhiBinWidth ;
  m_gctEtSumPhiOffset = gctEtSumPhiBinOffset * m_gctEtSumPhiBinWidth ;
  m_gctHtSumPhiOffset = gctHtSumPhiBinOffset * m_gctHtSumPhiBinWidth ;
}

// L1CaloGeometry::L1CaloGeometry(const L1CaloGeometry& rhs)
// {
//    // do actual copying here;
// }

L1CaloGeometry::~L1CaloGeometry()
{
}

//
// assignment operators
//
// const L1CaloGeometry& L1CaloGeometry::operator=(const L1CaloGeometry& rhs)
// {
//   //An exception safe implementation is
//   L1CaloGeometry temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

double
L1CaloGeometry::globalEtaBinCenter( unsigned int globalEtaIndex ) const
{
   int etaIndex ;
   double etaSign = 1. ;
   if( globalEtaIndex < m_etaBinsPerHalf )
   {
      etaIndex = m_etaBinsPerHalf - globalEtaIndex - 1 ;
      etaSign = -1. ;
   }
   else
   {
      etaIndex = globalEtaIndex - m_etaBinsPerHalf ;
   }

   return 0.5 * etaSign *
      ( m_gctEtaBinBoundaries[ etaIndex ] +
        m_gctEtaBinBoundaries[ etaIndex + 1 ] ) ;
}

double
L1CaloGeometry::globalEtaBinLowEdge( unsigned int globalEtaIndex ) const
{
   int etaIndex ;
   double etaSign = 1. ;
   if( globalEtaIndex < m_etaBinsPerHalf )
   {
      etaIndex = m_etaBinsPerHalf - globalEtaIndex - 1 ;
      etaSign = -1. ;
   }
   else
   {
      etaIndex = globalEtaIndex - m_etaBinsPerHalf ;
   }

   return ( etaSign > 0. ? 
	    m_gctEtaBinBoundaries[ etaIndex ] :
	    -m_gctEtaBinBoundaries[ etaIndex + 1 ] ) ;
}

double
L1CaloGeometry::globalEtaBinHighEdge( unsigned int globalEtaIndex ) const
{
   int etaIndex ;
   double etaSign = 1. ;
   if( globalEtaIndex < m_etaBinsPerHalf )
   {
      etaIndex = m_etaBinsPerHalf - globalEtaIndex - 1 ;
      etaSign = -1. ;
   }
   else
   {
      etaIndex = globalEtaIndex - m_etaBinsPerHalf ;
   }

   return ( etaSign > 0. ? 
	    m_gctEtaBinBoundaries[ etaIndex + 1 ] :
	    -m_gctEtaBinBoundaries[ etaIndex ] ) ;
}

double
L1CaloGeometry::etaBinCenter( unsigned int etaIndex,
			      bool central ) const
{
   // Central/tau jets and EM have etaIndex = 0-6 for eta = 0-3.
   // Forward jets have etaIndex = 0-3 for eta = 3-5.
   double etaSign = 1. ;

   // Check sign BEFORE shifting forward jet bin index.
   if( etaIndex >= m_etaSignBitOffset )
   {
      etaSign = -1. ;
      etaIndex -= m_etaSignBitOffset ;
   }

   // Shift forward jet bin index AFTER checking sign bit.
   if( !central )
   {
      etaIndex += m_numberGctCentralEtaBinsPerHalf ;
   }

   return 0.5 * etaSign *
      ( m_gctEtaBinBoundaries[ etaIndex ] +
        m_gctEtaBinBoundaries[ etaIndex + 1 ] ) ;
}

double
L1CaloGeometry::etaBinLowEdge( unsigned int etaIndex,
			       bool central ) const
{
   // Central/tau jets and EM have etaIndex = 0-6 for eta = 0-3.
   // Forward jets have etaIndex = 0-3 for eta = 3-5.
   double etaSign = 1. ;

   // Check sign BEFORE shifting forward jet bin index.
   if( etaIndex >= m_etaSignBitOffset )
   {
      etaSign = -1. ;
      etaIndex -= m_etaSignBitOffset ;
   }

   // Shift forward jet bin index AFTER checking sign bit.
   if( !central )
   {
      etaIndex += m_numberGctCentralEtaBinsPerHalf ;
   }

   return ( etaSign > 0. ? 
	    m_gctEtaBinBoundaries[ etaIndex ] :
	    -m_gctEtaBinBoundaries[ etaIndex + 1 ] ) ;
}

double
L1CaloGeometry::etaBinHighEdge( unsigned int etaIndex,
				bool central ) const
{
   // Central/tau jets and EM have etaIndex = 0-6 for eta = 0-3.
   // Forward jets have etaIndex = 0-3 for eta = 3-5.
   double etaSign = 1. ;

   // Check sign BEFORE shifting forward jet bin index.
   if( etaIndex >= m_etaSignBitOffset )
   {
      etaSign = -1. ;
      etaIndex -= m_etaSignBitOffset ;
   }

   // Shift forward jet bin index AFTER checking sign bit.
   if( !central )
   {
      etaIndex += m_numberGctCentralEtaBinsPerHalf ;
   }

   return ( etaSign > 0. ? 
	    m_gctEtaBinBoundaries[ etaIndex + 1 ] :
	    -m_gctEtaBinBoundaries[ etaIndex ] ) ;
}

double
L1CaloGeometry::emJetPhiBinCenter( unsigned int phiIndex ) const
{
   return ( ( double ) phiIndex + 0.5 ) * m_gctEmJetPhiBinWidth +
      m_gctEmJetPhiOffset ;
}

double
L1CaloGeometry::emJetPhiBinLowEdge( unsigned int phiIndex ) const
{
   return ( ( double ) phiIndex ) * m_gctEmJetPhiBinWidth +
      m_gctEmJetPhiOffset ;
}

double
L1CaloGeometry::emJetPhiBinHighEdge( unsigned int phiIndex ) const
{
   return ( ( double ) phiIndex + 1. ) * m_gctEmJetPhiBinWidth +
      m_gctEmJetPhiOffset ;
}

double
L1CaloGeometry::etSumPhiBinCenter( unsigned int phiIndex ) const
{
   return ( ( double ) phiIndex + 0.5 ) * m_gctEtSumPhiBinWidth +
      m_gctEtSumPhiOffset ;
}

double
L1CaloGeometry::etSumPhiBinLowEdge( unsigned int phiIndex ) const
{
   return ( ( double ) phiIndex ) * m_gctEtSumPhiBinWidth +
      m_gctEtSumPhiOffset ;
}

double
L1CaloGeometry::etSumPhiBinHighEdge( unsigned int phiIndex ) const
{
   return ( ( double ) phiIndex + 1. ) * m_gctEtSumPhiBinWidth +
      m_gctEtSumPhiOffset ;
}

double
L1CaloGeometry::htSumPhiBinCenter( unsigned int phiIndex ) const
{
  if( m_version == kOrig )
    {
      return ( ( double ) phiIndex + 0.5 ) * m_gctEtSumPhiBinWidth * 4. +
	m_gctEtSumPhiOffset ;
    }
  else
    {
      return ( ( double ) phiIndex + 0.5 ) * m_gctHtSumPhiBinWidth +
	m_gctHtSumPhiOffset ;
    }
}

double
L1CaloGeometry::htSumPhiBinLowEdge( unsigned int phiIndex ) const
{
  if( m_version == kOrig )
    {
      return ( ( double ) phiIndex ) * m_gctEtSumPhiBinWidth * 4. +
	m_gctEtSumPhiOffset ;
    }
  else
    {
      return ( ( double ) phiIndex ) * m_gctHtSumPhiBinWidth +
	m_gctHtSumPhiOffset ;
    }
}

double
L1CaloGeometry::htSumPhiBinHighEdge( unsigned int phiIndex ) const
{
  if( m_version == kOrig )
    {
      return ( ( double ) phiIndex + 1. ) * m_gctEtSumPhiBinWidth * 4. +
	m_gctEtSumPhiOffset ;
    }
  else
    {
      return ( ( double ) phiIndex + 1. ) * m_gctHtSumPhiBinWidth +
	m_gctHtSumPhiOffset ;
    }
}

unsigned int
L1CaloGeometry::etaIndex( const double& etaValue ) const
{
   unsigned int etaIndex = 0 ;

   for( unsigned int i = 0 ; i < m_numberGctCentralEtaBinsPerHalf ; ++i )
   {
      if( fabs( etaValue ) >= m_gctEtaBinBoundaries[ i ] )
      {
	 etaIndex = i ;
      }
   }

   for( unsigned int i = 0 ; i < m_numberGctForwardEtaBinsPerHalf ; ++i )
   {
      if( fabs( etaValue ) >=
	  m_gctEtaBinBoundaries[ i + m_numberGctCentralEtaBinsPerHalf ] )
      {
	 etaIndex = i ;
      }
   }

   if( etaValue < 0. )
   {
      etaIndex += m_etaSignBitOffset ;
   }

   return etaIndex ;
}

unsigned int
L1CaloGeometry::globalEtaIndex( const double& etaValue ) const
{
   unsigned int etaIndex = 0 ;

   if( etaValue < 0. )
   {
      for( unsigned int i = m_etaBinsPerHalf ; i > 0 ; --i )
      {
	 if( fabs( etaValue ) < m_gctEtaBinBoundaries[ i ] )
	 {
	    etaIndex = m_etaBinsPerHalf - i ;
	 }
      }
   }
   else
   {
      for( unsigned int i = 0 ; i < m_etaBinsPerHalf ; ++i )
      {
	 if( etaValue >= m_gctEtaBinBoundaries[ i ] )
	 {
	    etaIndex = i + m_etaBinsPerHalf ;
	 }
      }
   }

   return etaIndex ;
}

unsigned int
L1CaloGeometry::emJetPhiIndex( const double& phiValue ) const
{
   double phiAdjusted = phiValue - m_gctEmJetPhiOffset ;

   // Check phiValue is between m_gctEmJetPhiOffset and m_gctEmJetPhiOffset+2pi
   if( phiAdjusted < 0. )
   {
      do
      {
         phiAdjusted += 2. * M_PI ;
      }
      while( phiAdjusted < 0. ) ;
   }
   else if( phiAdjusted > 2. * M_PI )
   {
      do
      {
         phiAdjusted -= 2. * M_PI ;
      }
      while( phiAdjusted > 2. * M_PI ) ;
   }

   return ( ( int ) ( phiAdjusted / m_gctEmJetPhiBinWidth ) ) ;
}

unsigned int
L1CaloGeometry::etSumPhiIndex( const double& phiValue ) const
{
   double phiAdjusted = phiValue - m_gctEtSumPhiOffset ;

   // Check phiValue is between m_gctEtSumPhiOffset and m_gctEtSumPhiOffset+2pi
   if( phiAdjusted < 0. )
   {
      do
      {
         phiAdjusted += 2. * M_PI ;
      }
      while( phiAdjusted < 0. ) ;
   }
   else if( phiAdjusted > 2. * M_PI )
   {
      do
      {
         phiAdjusted -= 2. * M_PI ;
      }
      while( phiAdjusted > 2. * M_PI ) ;
   }

   return ( ( int ) ( phiAdjusted / m_gctEtSumPhiBinWidth ) ) ;
}

unsigned int
L1CaloGeometry::htSumPhiIndex( const double& phiValue ) const
{
   double phiAdjusted = phiValue - m_gctEtSumPhiOffset ;

   // Check phiValue is between m_gctEtSumPhiOffset and m_gctEtSumPhiOffset+2pi
   if( phiAdjusted < 0. )
   {
      do
      {
         phiAdjusted += 2. * M_PI ;
      }
      while( phiAdjusted < 0. ) ;
   }
   else if( phiAdjusted > 2. * M_PI )
   {
      do
      {
         phiAdjusted -= 2. * M_PI ;
      }
      while( phiAdjusted > 2. * M_PI ) ;
   }

   if( m_version == kOrig )
     {
       return ( ( int ) ( phiAdjusted / ( m_gctEtSumPhiBinWidth * 4. ) ) ) ;
     }
   else
     {
       return ( ( int ) ( phiAdjusted / m_gctHtSumPhiBinWidth ) ) ;
     }
}

unsigned int
L1CaloGeometry::numberGctHtSumPhiBins() const
{
  if( m_version == kOrig )
    {
      return m_numberGctEtSumPhiBins / 4 ;
    }
  else
    {
      return m_numberGctHtSumPhiBins ;
    }
}

std::ostream& operator << ( std::ostream& os, const L1CaloGeometry& obj )
{
   os << "L1CaloGeometry:" << std::endl ;

   os << "Central/tau eta bins: low / center / high" << std::endl ;
   for( unsigned int i = 0 ; i < obj.numberGctCentralEtaBinsPerHalf() ; ++i )
     {
       os << "  bin " << i << ": "
	  << obj.etaBinLowEdge( i ) << " / "
	  << obj.etaBinCenter( i ) << " / "
	  << obj.etaBinHighEdge( i )
	  << std::endl ;
     }

   os << "Forward eta bins: low / center / high" << std::endl ;
   for( unsigned int i = 0 ; i < obj.numberGctForwardEtaBinsPerHalf() ; ++i )
     {
       os << "  bin " << i << ": "
	  << obj.etaBinLowEdge( i, false ) << " / "
	  << obj.etaBinCenter( i, false ) << " / "
	  << obj.etaBinHighEdge( i, false )
	  << std::endl ;
     }

   os << "Global eta bins: low / center / high" << std::endl ;
   for( unsigned int i = 0 ; i < obj.numberGctCentralEtaBinsPerHalf() +
	  obj.numberGctForwardEtaBinsPerHalf() ; ++i )
     {
       os << "  bin " << i << ": "
	  << obj.globalEtaBinLowEdge( i ) << " / "
	  << obj.globalEtaBinCenter( i ) << " / "
	  << obj.globalEtaBinHighEdge( i )
	  << std::endl ;
     }

   os << "EM/jet phi bins: low / center / high" << std::endl ;
   for( unsigned int i = 0 ; i < obj.numberGctEmJetPhiBins() ; ++i )
     {
       os << "  bin " << i << ": "
	  << obj.emJetPhiBinLowEdge( i ) << " / "
	  << obj.emJetPhiBinCenter( i ) << " / "
	  << obj.emJetPhiBinHighEdge( i )
	  << std::endl ;
     }

   os << "Et sum phi bins: low / center / high" << std::endl ;
   for( unsigned int i = 0 ; i < obj.numberGctEtSumPhiBins() ; ++i )
     {
       os << "  bin " << i << ": "
	  << obj.etSumPhiBinLowEdge( i ) << " / "
	  << obj.etSumPhiBinCenter( i ) << " / "
	  << obj.etSumPhiBinHighEdge( i )
	  << std::endl ;
     }

   os << "Ht sum phi bins: low / center / high" << std::endl ;
   for( unsigned int i = 0 ; i < obj.numberGctHtSumPhiBins() ; ++i )
     {
       os << "  bin " << i << ": "
	  << obj.htSumPhiBinLowEdge( i ) << " / "
	  << obj.htSumPhiBinCenter( i ) << " / "
	  << obj.htSumPhiBinHighEdge( i )
	  << std::endl ;
     }

   return os ;
}

//
// static member functions
//

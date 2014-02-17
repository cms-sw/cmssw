#ifndef L1Geometry_L1CaloGeometry_h
#define L1Geometry_L1CaloGeometry_h
// -*- C++ -*-
//
// Package:     L1Geometry
// Class  :     L1CaloGeometry
// 
/**\class L1CaloGeometry L1CaloGeometry.h L1TriggerConfig/L1Geometry/interface/L1CaloGeometry.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Mon Oct 23 21:52:29 EDT 2006
// $Id: L1CaloGeometry.h,v 1.4 2009/09/28 22:59:12 wsun Exp $
//

// system include files
#include <vector>
#include <ostream>

// user include files
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

// forward declarations

class L1CaloGeometry
{

   public:
/*       static const unsigned int kNumberGctEmJetPhiBins = 18 ; */
/*       static const unsigned int kNumberGctEtSumPhiBins = 72 ; */
/*       static const unsigned int kNumberGctCentralEtaBinsPerHalf = 7 ; */
/*       static const unsigned int kNumberGctForwardEtaBinsPerHalf = 4 ; */

      // calo sign bit is the 4th bit
/*       static const unsigned int kEtaSignBitOffset = 8 ; */

      enum Versions{ kOrig, kAddedMHTPhi, kNumVersions } ;

      L1CaloGeometry();
      L1CaloGeometry( unsigned int numberGctEmJetPhiBins,
		      double gctEmJetPhiBinOffset, // -0.5 bins usually
		      unsigned int numberGctEtSumPhiBins,
		      double gctEtSumPhiBinOffset, // 0 bins usually
		      unsigned int numberGctHtSumPhiBins,
		      double gctHtSumPhiBinOffset, // 0 bins usually
		      unsigned int numberGctCentralEtaBinsPerHalf,
		      unsigned int numberGctForwardEtaBinsPerHalf,
		      unsigned int etaSignBitOffset,
		      const std::vector< double >& gctEtaBinBoundaries ) ;
      virtual ~L1CaloGeometry();

      // ---------- const member functions ---------------------

      unsigned int version() const { return m_version ; }

      // Central/tau jets and EM have etaIndex = 0-6 for eta = 0.0-3.0
      // Forward jets have etaIndex = 0-3 for eta = 3.0-5.0
      double etaBinCenter( unsigned int etaIndex,
			   bool central = true ) const ;
      double etaBinLowEdge( unsigned int etaIndex,
			    bool central = true ) const ;
      double etaBinHighEdge( unsigned int etaIndex,
			     bool central = true ) const ;

      // Global index = 0-21
      double globalEtaBinCenter( unsigned int globalEtaIndex ) const ;
      double globalEtaBinLowEdge( unsigned int globalEtaIndex ) const ;
      double globalEtaBinHighEdge( unsigned int globalEtaIndex ) const ;

      // Eta index of L1CaloRegionDetId is global index 0-21.
      double etaBinCenter( const L1CaloRegionDetId& detId ) const
      { return globalEtaBinCenter( detId.ieta() ) ; }
      double etaBinLowEdge( const L1CaloRegionDetId& detId ) const
      { return globalEtaBinLowEdge( detId.ieta() ) ; }
      double etaBinHighEdge( const L1CaloRegionDetId& detId ) const
      { return globalEtaBinHighEdge( detId.ieta() ) ; }

      double emJetPhiBinCenter( unsigned int phiIndex ) const ;
      double emJetPhiBinLowEdge( unsigned int phiIndex ) const ;
      double emJetPhiBinHighEdge( unsigned int phiIndex ) const ;

      double emJetPhiBinCenter( const L1CaloRegionDetId& detId ) const
      { return emJetPhiBinCenter( detId.iphi() ) ; }
      double emJetPhiBinLowEdge( const L1CaloRegionDetId& detId ) const
      { return emJetPhiBinLowEdge( detId.iphi() ) ; }
      double emJetPhiBinHighEdge( const L1CaloRegionDetId& detId ) const
      { return emJetPhiBinHighEdge( detId.iphi() ) ; }

      double etSumPhiBinCenter( unsigned int phiIndex ) const ;
      double etSumPhiBinLowEdge( unsigned int phiIndex ) const ;
      double etSumPhiBinHighEdge( unsigned int phiIndex ) const ;

      double htSumPhiBinCenter( unsigned int phiIndex ) const ;
      double htSumPhiBinLowEdge( unsigned int phiIndex ) const ;
      double htSumPhiBinHighEdge( unsigned int phiIndex ) const ;

      unsigned int etaIndex( const double& etaValue ) const ; // 0-6 or 0-3
      unsigned int globalEtaIndex( const double& etaValue ) const ; // 0-21
      unsigned int emJetPhiIndex( const double& phiValue ) const ;
      unsigned int etSumPhiIndex( const double& phiValue ) const ;
      unsigned int htSumPhiIndex( const double& phiValue ) const ;

      unsigned int numberGctEmJetPhiBins() const
      { return m_numberGctEmJetPhiBins ; }
      unsigned int numberGctEtSumPhiBins() const
      { return m_numberGctEtSumPhiBins ; }
      unsigned int numberGctHtSumPhiBins() const ;
      unsigned int numberGctCentralEtaBinsPerHalf() const
      { return m_numberGctCentralEtaBinsPerHalf ; }
      unsigned int numberGctForwardEtaBinsPerHalf() const
      { return m_numberGctForwardEtaBinsPerHalf ; }
      unsigned int etaSignBitOffset() const
      { return m_etaBinsPerHalf ; }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      //L1CaloGeometry(const L1CaloGeometry&); // stop default

      //const L1CaloGeometry& operator=(const L1CaloGeometry&); // stop default

      // ---------- member data --------------------------------

      unsigned int m_version ;

      unsigned int m_numberGctEmJetPhiBins ;
      unsigned int m_numberGctEtSumPhiBins ;
      unsigned int m_numberGctHtSumPhiBins ;
      unsigned int m_numberGctCentralEtaBinsPerHalf ;
      unsigned int m_numberGctForwardEtaBinsPerHalf ;
      unsigned int m_etaSignBitOffset ;
      std::vector< double > m_gctEtaBinBoundaries ;

      unsigned int m_etaBinsPerHalf ;

      // Calo phi bins are uniform.
      double m_gctEmJetPhiBinWidth ;
      double m_gctEtSumPhiBinWidth ;
      double m_gctHtSumPhiBinWidth ;
      double m_gctEmJetPhiOffset ;
      double m_gctEtSumPhiOffset ;
      double m_gctHtSumPhiOffset ;
};

std::ostream& operator << ( std::ostream& os, const L1CaloGeometry& obj ) ;

#endif

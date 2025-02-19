#ifndef Alignment_MuonAlignment_MuonAlignmentOutputXML_h
#define Alignment_MuonAlignment_MuonAlignmentOutputXML_h
// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentOutputXML
// 
/**\class MuonAlignmentOutputXML MuonAlignmentOutputXML.h Alignment/MuonAlignment/interface/MuonAlignmentOutputXML.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Fri Mar 14 18:02:28 CDT 2008
// $Id: MuonAlignmentOutputXML.h,v 1.4 2011/06/07 19:28:47 khotilov Exp $
//

// system include files
#include <fstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

// user include files
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"

// forward declarations

class MuonAlignmentOutputXML {

   public:
      MuonAlignmentOutputXML(const edm::ParameterSet &iConfig);
      virtual ~MuonAlignmentOutputXML();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      void write(AlignableMuon *alignableMuon, const edm::EventSetup &iSetup) const;

   private:
      MuonAlignmentOutputXML(const MuonAlignmentOutputXML&); // stop default

      const MuonAlignmentOutputXML& operator=(const MuonAlignmentOutputXML&); // stop default

      void writeComponents(align::Alignables &alignables, align::Alignables &ideals,
			   std::map<align::ID, CLHEP::HepSymMatrix>& errors, std::ofstream &outputFile, bool DT) const;

      // ---------- member data --------------------------------
      std::string m_fileName;
      int m_relativeto;
      bool m_survey, m_rawIds, m_eulerAngles;
      int m_precision;
      bool m_suppressDTBarrel, m_suppressDTWheels, m_suppressDTStations, m_suppressDTChambers, m_suppressDTSuperLayers, m_suppressDTLayers;
      bool m_suppressCSCEndcaps, m_suppressCSCStations, m_suppressCSCRings, m_suppressCSCChambers, m_suppressCSCLayers;
};

#endif

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
// $Id: MuonAlignmentOutputXML.h,v 1.3 2008/05/17 18:10:19 pivarski Exp $
//

// system include files
#include <fstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

// user include files
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "CondFormats/Alignment/interface/AlignTransformErrorExtended.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
// forward declarations
class AlignableObjectId;

class MuonAlignmentOutputXML {
public:
  MuonAlignmentOutputXML(const edm::ParameterSet &iConfig,
                         const DTGeometry *dtGeometry,
                         const CSCGeometry *cscGeometry,
                         const GEMGeometry *gemGeometry);
  virtual ~MuonAlignmentOutputXML();

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  void write(AlignableMuon *alignableMuon) const;

  MuonAlignmentOutputXML(const MuonAlignmentOutputXML &) = delete;  // stop default

  const MuonAlignmentOutputXML &operator=(const MuonAlignmentOutputXML &) = delete;  // stop default

private:
  enum { doDT, doCSC, doGEM };
  void writeComponents(align::Alignables &alignables,
                       align::Alignables &ideals,
                       std::map<align::ID, CLHEP::HepSymMatrix> &errors,
                       std::ofstream &outputFile,
                       const int doDet,
                       const AlignableObjectId &) const;

  // ---------- member data --------------------------------
  std::string m_fileName;
  int m_relativeto;
  bool m_survey, m_rawIds, m_eulerAngles;
  int m_precision;
  bool m_suppressDTBarrel, m_suppressDTWheels, m_suppressDTStations, m_suppressDTChambers, m_suppressDTSuperLayers,
      m_suppressDTLayers;
  bool m_suppressCSCEndcaps, m_suppressCSCStations, m_suppressCSCRings, m_suppressCSCChambers, m_suppressCSCLayers;
  bool m_suppressGEMEndcaps, m_suppressGEMStations, m_suppressGEMRings, m_suppressGEMSuperChambers,
      m_suppressGEMChambers, m_suppressGEMEtaPartitions;

  const DTGeometry *dtGeometry_;
  const CSCGeometry *cscGeometry_;
  const GEMGeometry *gemGeometry_;
};

#endif

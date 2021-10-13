#ifndef Alignment_MuonAlignment_MuonAlignmentInputDB_h
#define Alignment_MuonAlignment_MuonAlignmentInputDB_h
// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputDB
//
/**\class MuonAlignmentInputDB MuonAlignmentInputDB.h Alignment/MuonAlignment/interface/MuonAlignmentInputDB.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 17:30:40 CST 2008
// $Id: MuonAlignmentInputDB.h,v 1.1 2008/03/15 20:26:46 pivarski Exp $
//

#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentErrorRcd.h"

class MuonAlignmentInputDB : public MuonAlignmentInputMethod {
public:
  MuonAlignmentInputDB(edm::ConsumesCollector iC);
  MuonAlignmentInputDB(std::string dtLabel,
                       std::string cscLabel,
                       std::string gemLabel,
                       std::string idealLabel,
                       bool getAPEs,
                       edm::ConsumesCollector iC);
  ~MuonAlignmentInputDB() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const override;

  MuonAlignmentInputDB(const MuonAlignmentInputDB &) = delete;  // stop default

  const MuonAlignmentInputDB &operator=(const MuonAlignmentInputDB &) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  const std::string m_dtLabel, m_cscLabel, m_gemLabel, idealGeometryLabel;
  const bool m_getAPEs;

  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken_;
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemGeomToken_;
  const edm::ESGetToken<Alignments, DTAlignmentRcd> dtAliToken_;
  const edm::ESGetToken<Alignments, CSCAlignmentRcd> cscAliToken_;
  const edm::ESGetToken<Alignments, GEMAlignmentRcd> gemAliToken_;
  const edm::ESGetToken<AlignmentErrorsExtended, DTAlignmentErrorExtendedRcd> dtAliErrToken_;
  const edm::ESGetToken<AlignmentErrorsExtended, CSCAlignmentErrorExtendedRcd> cscAliErrToken_;
  const edm::ESGetToken<AlignmentErrorsExtended, GEMAlignmentErrorExtendedRcd> gemAliErrToken_;
  const edm::ESGetToken<Alignments, GlobalPositionRcd> gprToken_;
};

#endif

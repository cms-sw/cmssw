#ifndef Alignment_MuonAlignment_MuonAlignmentInputSurveyDB_h
#define Alignment_MuonAlignment_MuonAlignmentInputSurveyDB_h
// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputSurveyDB
//
/**\class MuonAlignmentInputSurveyDB MuonAlignmentInputSurveyDB.h Alignment/MuonAlignment/interface/MuonAlignmentInputSurveyDB.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Fri Mar  7 16:13:19 CST 2008
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorExtendedRcd.h"
// forward declarations

class MuonAlignmentInputSurveyDB : public MuonAlignmentInputMethod {
public:
  MuonAlignmentInputSurveyDB(edm::ConsumesCollector iC);
  MuonAlignmentInputSurveyDB(std::string dtLabel,
                             std::string cscLabel,
                             std::string idealLabel,
                             edm::ConsumesCollector iC);
  ~MuonAlignmentInputSurveyDB() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  AlignableMuon* newAlignableMuon(const edm::EventSetup& iSetup) const override;

  MuonAlignmentInputSurveyDB(const MuonAlignmentInputSurveyDB&) = delete;  // stop default

  const MuonAlignmentInputSurveyDB& operator=(const MuonAlignmentInputSurveyDB&) = delete;  // stop default

private:
  void addSurveyInfo_(Alignable* ali,
                      unsigned int* theSurveyIndex,
                      const Alignments* theSurveyValues,
                      const SurveyErrors* theSurveyErrors) const;

  // ---------- member data --------------------------------
  const std::string m_dtLabel, m_cscLabel, idealGeometryLabel;

  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken_;
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemGeomToken_;
  const edm::ESGetToken<Alignments, DTSurveyRcd> dtSurveyToken_;
  const edm::ESGetToken<SurveyErrors, DTSurveyErrorExtendedRcd> dtSurvErrorToken_;
  const edm::ESGetToken<Alignments, CSCSurveyRcd> cscSurveyToken_;
  const edm::ESGetToken<SurveyErrors, CSCSurveyErrorExtendedRcd> cscSurvErrorToken_;
};

#endif

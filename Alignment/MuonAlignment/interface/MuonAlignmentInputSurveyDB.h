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
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"

// forward declarations

class MuonAlignmentInputSurveyDB : public MuonAlignmentInputMethod {
public:
  MuonAlignmentInputSurveyDB(const DTGeometry* dtGeometry,
                             const CSCGeometry* cscGeometry,
                             const GEMGeometry* gemGeometry,
                             const Alignments* dtSurvey,
                             const Alignments* cscSurvey,
                             const SurveyErrors* dtSurveyError,
                             const SurveyErrors* cscSurveyError);
  ~MuonAlignmentInputSurveyDB() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  AlignableMuon* newAlignableMuon() const override;

  MuonAlignmentInputSurveyDB(const MuonAlignmentInputSurveyDB&) = delete;  // stop default

  const MuonAlignmentInputSurveyDB& operator=(const MuonAlignmentInputSurveyDB&) = delete;  // stop default

private:
  void addSurveyInfo_(Alignable* ali,
                      unsigned int* theSurveyIndex,
                      const Alignments* theSurveyValues,
                      const SurveyErrors* theSurveyErrors) const;

  // ---------- member data --------------------------------
  const DTGeometry* dtGeometry_;
  const CSCGeometry* cscGeometry_;
  const GEMGeometry* gemGeometry_;
  const Alignments* dtSurvey_;
  const Alignments* cscSurvey_;
  const SurveyErrors* dtSurveyError_;
  const SurveyErrors* cscSurveyError_;
};

#endif

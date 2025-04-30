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

// system include files

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"

// forward declarations

class MuonAlignmentInputDB : public MuonAlignmentInputMethod {
public:
  MuonAlignmentInputDB(const DTGeometry* dtGeometry,
                       const CSCGeometry* cscGeometry,
                       const GEMGeometry* gemGeometry,
                       const Alignments* dtAlignments,
                       const Alignments* cscAlignments,
                       const Alignments* gemAlignments,
                       const Alignments* globalPositionRcd);
  MuonAlignmentInputDB(const DTGeometry* dtGeometry,
                       const CSCGeometry* cscGeometry,
                       const GEMGeometry* gemGeometry,
                       const Alignments* dtAlignments,
                       const Alignments* cscAlignments,
                       const Alignments* gemAlignments,
                       const AlignmentErrorsExtended* dtAlignmentErrorsExtended,
                       const AlignmentErrorsExtended* cscAlignmentErrorsExtended,
                       const AlignmentErrorsExtended* gemAlignmentErrorsExtended,
                       const Alignments* globalPositionRcd);
  ~MuonAlignmentInputDB() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  AlignableMuon* newAlignableMuon() const override;

  MuonAlignmentInputDB(const MuonAlignmentInputDB&) = delete;  // stop default

  const MuonAlignmentInputDB& operator=(const MuonAlignmentInputDB&) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  const DTGeometry* dtGeometry_;
  const CSCGeometry* cscGeometry_;
  const GEMGeometry* gemGeometry_;
  const Alignments* dtAlignments_;
  const Alignments* cscAlignments_;
  const Alignments* gemAlignments_;
  const AlignmentErrorsExtended* dtAlignmentErrorsExtended_;
  const AlignmentErrorsExtended* cscAlignmentErrorsExtended_;
  const AlignmentErrorsExtended* gemAlignmentErrorsExtended_;
  const Alignments* globalPositionRcd_;

  const bool m_getAPEs;
};

#endif
-- dummy change --

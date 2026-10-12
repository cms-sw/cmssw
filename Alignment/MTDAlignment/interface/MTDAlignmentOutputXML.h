#ifndef Alignment_MTDAlignment_MTDAlignmentOutputXML_h
#define Alignment_MTDAlignment_MTDAlignmentOutputXML_h
// -*- C++ -*-
//
// Package:     MTDAlignment
// Class  :     MTDAlignmentOutputXML
//
//

// system include files
#include <fstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

// user include files
#include "Alignment/MTDAlignment/interface/AlignableMTD.h"
#include "CondFormats/Alignment/interface/AlignTransformErrorExtended.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
// forward declarations
class AlignableObjectId;

class MTDAlignmentOutputXML {
public:
  MTDAlignmentOutputXML(const edm::ParameterSet &iConfig, const MTDGeometry *mtdGeometry);
  virtual ~MTDAlignmentOutputXML();

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  void write(AlignableMTD *alignableMTD) const;

  MTDAlignmentOutputXML(const MTDAlignmentOutputXML &) = delete;  // stop default

  const MTDAlignmentOutputXML &operator=(const MTDAlignmentOutputXML &) = delete;  // stop default

private:
  enum { doBTL, doETL };
  void writeComponents(align::Alignables &alignables,
                       align::Alignables &ideals,
                       std::map<align::ID, CLHEP::HepSymMatrix> &errors,
                       std::ofstream &outputFile,
                       const int doDet,
                       const AlignableObjectId &) const;

  // ---------- member data --------------------------------
  std::string m_fileName;
  int m_relativeto;
  bool m_rawIds, m_eulerAngles;
  int m_precision;

  const MTDGeometry *mtdGeometry_;
};

#endif

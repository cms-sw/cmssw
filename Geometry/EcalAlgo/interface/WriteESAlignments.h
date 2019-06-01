#ifndef SOMEPACKAGE_WRITEESALIGNMENTS_H
#define SOMEPACKAGE_WRITEESALIGNMENTS_H 1

namespace edm {
  class ConsumesCollector;
  class EventSetup;
}  // namespace edm

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class WriteESAlignments {
public:
  typedef Alignments* AliPtr;
  typedef std::vector<AlignTransform> AliVec;

  typedef AlignTransform::Translation Trl;
  typedef AlignTransform::Rotation Rot;

  typedef std::vector<double> DVec;

  static const unsigned int k_nA;

  WriteESAlignments(edm::ConsumesCollector&& cc);

  void writeAlignments(const edm::EventSetup& eventSetup,
                       const DVec& alphaVec,
                       const DVec& betaVec,
                       const DVec& gammaVec,
                       const DVec& xtranslVec,
                       const DVec& ytranslVec,
                       const DVec& ztranslVec);

private:
  void convert(const edm::EventSetup& eS,
               const DVec& a,
               const DVec& b,
               const DVec& g,
               const DVec& x,
               const DVec& y,
               const DVec& z,
               AliVec& va);

  void write(AliPtr aliPtr);

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  edm::ESGetToken<Alignments, ESAlignmentRcd> alignmentToken_;
};

#endif

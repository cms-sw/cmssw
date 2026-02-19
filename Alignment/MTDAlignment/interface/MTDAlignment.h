#ifndef Alignment_MTDAlignment_MTDAlignment_H
#define Alignment_MTDAlignment_MTDAlignment_H

/** \class MTDAlignment
 *  The MTDAlignment helper class for alignment jobs
 *
 *  $Date: 2024/12/07 19:28:47 $
 *  $Revision: 1.14 $
 *  \author Pablo martinez Ruiz del Arbol - IFCA
 */

#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/MTDAlignment/interface/AlignableMTD.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/MTDAlignment/interface/MTDAlignmentInputMethod.h"
#include "Geometry/Records/interface/MTDGeometryRecord.h"

class MTDAlignment {
public:
  MTDAlignment(const MTDGeometry* mtdGeometry);

  MTDAlignment(const edm::EventSetup& iSetup, const MTDAlignmentInputMethod& input);

  ~MTDAlignment() {
    delete theAlignableMTD;
    delete theAlignableNavigator;
  }

  AlignableMTD* getAlignableMTD() { return theAlignableMTD; }

  AlignableNavigator* getAlignableNavigator() { return theAlignableNavigator; }

  void moveAlignableLocalCoord(DetId&, align::Scalars&, align::Scalars&);
  void moveAlignableGlobalCoord(DetId&, align::Scalars&, align::Scalars&);

  void recursiveList(const align::Alignables& alignables, align::Alignables& theList);
  void recursiveMap(const align::Alignables& alignables, std::map<align::ID, Alignable*>& theMap);
  void recursiveStructureMap(const align::Alignables& alignables,
                             std::map<std::pair<align::StructureType, align::ID>, Alignable*>& theMap);

  //void writeXML(const edm::ParameterSet& iConfig,
  //              const MTDGeometry* mtdGeometryXML);

  void savetoDB();

private:
  void init();
  void recursiveCopySurveyToAlignment(Alignable* alignable);

  std::string theMTDAlignRecordName, theMTDErrorRecordName;

  const MTDGeometry* mtdGeometry_;

  align::Scalars displacements;

  align::Scalars rotations;

  AlignableMTD* theAlignableMTD;

  AlignableNavigator* theAlignableNavigator;
};

#endif  //MTDAlignment_H

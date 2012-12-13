#ifndef Alignment_MuonAlignment_MuonAlignmentInputMethod_h
#define Alignment_MuonAlignment_MuonAlignmentInputMethod_h

/**\class MuonAlignmentInputMethod

Base abstract class for muon alignment input

*/

//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 14:10:22 CST 2008
//
// $Id: MuonAlignmentInputMethod.h,v 1.2 2008/03/20 21:39:26 pivarski Exp $
//

#include <boost/shared_ptr.hpp>
#include "FWCore/Framework/interface/EventSetup.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"


class MuonAlignmentInputMethod
{
public:

  MuonAlignmentInputMethod() {}

  virtual ~MuonAlignmentInputMethod();

  virtual AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const;

protected:

  boost::shared_ptr<DTGeometry> idealDTGeometry(const edm::EventSetup &iSetup) const;
  boost::shared_ptr<CSCGeometry> idealCSCGeometry(const edm::EventSetup &iSetup) const;

private:

  MuonAlignmentInputMethod(const MuonAlignmentInputMethod&); // stop default

  const MuonAlignmentInputMethod& operator=(const MuonAlignmentInputMethod&); // stop default

};

#endif

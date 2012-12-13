//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 14:25:07 CST 2008
//
// $Id: MuonAlignmentInputMethod.cc,v 1.4 2010/05/14 21:07:59 pivarski Exp $
//

#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"


MuonAlignmentInputMethod::~MuonAlignmentInputMethod() {}


AlignableMuon *MuonAlignmentInputMethod::newAlignableMuon(const edm::EventSetup& iSetup) const
{
   boost::shared_ptr<DTGeometry> dtGeometry = idealDTGeometry(iSetup);
   boost::shared_ptr<CSCGeometry> cscGeometry = idealCSCGeometry(iSetup);
   return new AlignableMuon(&(*dtGeometry), &(*cscGeometry));
}


boost::shared_ptr<DTGeometry> MuonAlignmentInputMethod::idealDTGeometry(const edm::EventSetup& iSetup) const
{
   edm::ESTransientHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get(cpv);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);
   DTGeometryBuilderFromDDD DTGeometryBuilder;

   boost::shared_ptr<DTGeometry> boost_dtGeometry(new DTGeometry );
   DTGeometryBuilder.build(boost_dtGeometry, &(*cpv), *mdc);

   return boost_dtGeometry;
}


boost::shared_ptr<CSCGeometry> MuonAlignmentInputMethod::idealCSCGeometry(const edm::EventSetup& iSetup) const
{
   edm::ESTransientHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get(cpv);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);
   CSCGeometryBuilderFromDDD CSCGeometryBuilder;

   boost::shared_ptr<CSCGeometry> boost_cscGeometry(new CSCGeometry);
   CSCGeometryBuilder.build(boost_cscGeometry, &(*cpv), *mdc);

   return boost_cscGeometry;
}

// Framework
#include "FWCore/Framework/interface/ESHandle.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignment.h"

//__________________________________________________________________
//
TrackerAlignment::TrackerAlignment(const edm::EventSetup& setup)
    : theAlignRecordName("TrackerAlignmentRcd"), theErrorRecordName("TrackerAlignmentErrorExtendedRcd") {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  setup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::ESHandle<TrackerGeometry> trackerGeometry;
  setup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);
  theAlignableTracker = new AlignableTracker(&(*trackerGeometry), tTopo);
}

//__________________________________________________________________
//
TrackerAlignment::~TrackerAlignment(void) { delete theAlignableTracker; }

//__________________________________________________________________
//
void TrackerAlignment::moveAlignablePixelEndCaps(int rawid,
                                                 const align::Scalars& local_displacements,
                                                 const align::Scalars& local_rotations) {
  // Displace and rotate pixelEndCaps
  const align::Alignables& thePixelEndCapsAlignables = theAlignableTracker->pixelEndcapGeomDets();
  for (auto thePixelEndCapsAlignable : thePixelEndCapsAlignables) {
    // Get the raw ID of the associated GeomDet
    int id = thePixelEndCapsAlignable->geomDetId().rawId();

    // Select the given module
    if (id == rawid) {
      // Convert local to global diplacements
      align::LocalVector lvector(local_displacements.at(0), local_displacements.at(1), local_displacements.at(2));
      align::GlobalVector gvector = (thePixelEndCapsAlignable->surface()).toGlobal(lvector);

      // global displacement
      thePixelEndCapsAlignable->move(gvector);

      // local rotation
      thePixelEndCapsAlignable->rotateAroundLocalX(local_rotations.at(0));  // Local X axis rotation
      thePixelEndCapsAlignable->rotateAroundLocalY(local_rotations.at(1));  // Local Y axis rotation
      thePixelEndCapsAlignable->rotateAroundLocalZ(local_rotations.at(2));  // Local Z axis rotation
    }
  }
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignableEndCaps(int rawid,
                                            const align::Scalars& local_displacements,
                                            const align::Scalars& local_rotations) {
  // Displace and rotate EndCaps
  const align::Alignables& theEndCapsAlignables = theAlignableTracker->endcapGeomDets();
  for (auto theEndCapsAlignable : theEndCapsAlignables) {
    // Get the raw ID of the associated GeomDet
    int id = theEndCapsAlignable->geomDetId().rawId();

    // Select the given module
    if (id == rawid) {
      // Convert local to global diplacements
      align::LocalVector lvector(local_displacements.at(0), local_displacements.at(1), local_displacements.at(2));
      align::GlobalVector gvector = (theEndCapsAlignable->surface()).toGlobal(lvector);

      // global displacement
      theEndCapsAlignable->move(gvector);

      // local rotation
      theEndCapsAlignable->rotateAroundLocalX(local_rotations.at(0));  // Local X axis rotation
      theEndCapsAlignable->rotateAroundLocalY(local_rotations.at(1));  // Local Y axis rotation
      theEndCapsAlignable->rotateAroundLocalZ(local_rotations.at(2));  // Local Z axis rotation
    }
  }
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignablePixelHalfBarrels(int rawid,
                                                     const align::Scalars& local_displacements,
                                                     const align::Scalars& local_rotations) {
  // Displace and rotate PixelHalfBarrels
  const align::Alignables& thePixelHalfBarrelsAlignables = theAlignableTracker->pixelHalfBarrelGeomDets();
  for (auto thePixelHalfBarrelsAlignable : thePixelHalfBarrelsAlignables) {
    // Get the raw ID of the associated GeomDet
    int id = thePixelHalfBarrelsAlignable->geomDetId().rawId();

    // Select the given module
    if (id == rawid) {
      // Convert local to global diplacements
      align::LocalVector lvector(local_displacements.at(0), local_displacements.at(1), local_displacements.at(2));
      align::GlobalVector gvector = (thePixelHalfBarrelsAlignable->surface()).toGlobal(lvector);

      // global displacement
      thePixelHalfBarrelsAlignable->move(gvector);

      // local rotation
      thePixelHalfBarrelsAlignable->rotateAroundLocalX(local_rotations.at(0));  // Local X axis rotation
      thePixelHalfBarrelsAlignable->rotateAroundLocalY(local_rotations.at(1));  // Local Y axis rotation
      thePixelHalfBarrelsAlignable->rotateAroundLocalZ(local_rotations.at(2));  // Local Z axis rotation
    }
  }
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignableOuterHalfBarrels(int rawid,
                                                     const align::Scalars& local_displacements,
                                                     const align::Scalars& local_rotations) {
  // Displace and rotate OuterHalfBarrels
  const align::Alignables& theOuterHalfBarrelsAlignables = theAlignableTracker->outerBarrelGeomDets();
  for (auto theOuterHalfBarrelsAlignable : theOuterHalfBarrelsAlignables) {
    // Get the raw ID of the associated GeomDet
    int id = theOuterHalfBarrelsAlignable->geomDetId().rawId();

    // Select the given module
    if (id == rawid) {
      // Convert local to global diplacements
      align::LocalVector lvector(local_displacements.at(0), local_displacements.at(1), local_displacements.at(2));
      align::GlobalVector gvector = (theOuterHalfBarrelsAlignable->surface()).toGlobal(lvector);

      // global displacement
      theOuterHalfBarrelsAlignable->move(gvector);

      // local rotation
      theOuterHalfBarrelsAlignable->rotateAroundLocalX(local_rotations.at(0));  // Local X axis rotation
      theOuterHalfBarrelsAlignable->rotateAroundLocalY(local_rotations.at(1));  // Local Y axis rotation
      theOuterHalfBarrelsAlignable->rotateAroundLocalZ(local_rotations.at(2));  // Local Z axis rotation
    }
  }
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignableInnerHalfBarrels(int rawid,
                                                     const align::Scalars& local_displacements,
                                                     const align::Scalars& local_rotations) {
  // Displace and rotate InnerHalfBarrels
  const align::Alignables& theInnerHalfBarrelsAlignables = theAlignableTracker->innerBarrelGeomDets();
  for (auto theInnerHalfBarrelsAlignable : theInnerHalfBarrelsAlignables) {
    // Get the raw ID of the associated GeomDet
    int id = theInnerHalfBarrelsAlignable->geomDetId().rawId();

    // Select the given module
    if (id == rawid) {
      // Convert local to global diplacements
      align::LocalVector lvector(local_displacements.at(0), local_displacements.at(1), local_displacements.at(2));
      align::GlobalVector gvector = (theInnerHalfBarrelsAlignable->surface()).toGlobal(lvector);

      // global displacement
      theInnerHalfBarrelsAlignable->move(gvector);

      // local rotation
      theInnerHalfBarrelsAlignable->rotateAroundLocalX(local_rotations.at(0));  // Local X axis rotation
      theInnerHalfBarrelsAlignable->rotateAroundLocalY(local_rotations.at(1));  // Local Y axis rotation
      theInnerHalfBarrelsAlignable->rotateAroundLocalZ(local_rotations.at(2));  // Local Z axis rotation
    }
  }
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignableTIDs(int rawid,
                                         const align::Scalars& local_displacements,
                                         const align::Scalars& local_rotations) {
  // Displace and rotate TIDs
  const align::Alignables& theTIDsAlignables = theAlignableTracker->TIDGeomDets();
  for (auto theTIDsAlignable : theTIDsAlignables) {
    // Get the raw ID of the associated GeomDet
    int id = theTIDsAlignable->geomDetId().rawId();

    // Select the given module
    if (id == rawid) {
      // Convert local to global diplacements
      align::LocalVector lvector(local_displacements.at(0), local_displacements.at(1), local_displacements.at(2));
      align::GlobalVector gvector = (theTIDsAlignable->surface()).toGlobal(lvector);

      // global displacement
      theTIDsAlignable->move(gvector);

      // local rotation
      theTIDsAlignable->rotateAroundLocalX(local_rotations.at(0));  // Local X axis rotation
      theTIDsAlignable->rotateAroundLocalY(local_rotations.at(1));  // Local Y axis rotation
      theTIDsAlignable->rotateAroundLocalZ(local_rotations.at(2));  // Local Z axis rotation
    }
  }
}

//__________________________________________________________________
//
void TrackerAlignment::moveAlignableTIBTIDs(int rawId,
                                            const align::Scalars& globalDisplacements,
                                            const align::RotationType& backwardRotation,
                                            const align::RotationType& forwardRotation,
                                            bool toAndFro) {
  // Displace and rotate TIB and TID
  const align::Alignables& theTIBTIDAlignables = theAlignableTracker->TIBTIDGeomDets();
  for (auto theTIBTIDAlignable : theTIBTIDAlignables) {
    // Get the raw ID of the associated GeomDet
    int id = theTIBTIDAlignable->geomDetId().rawId();

    // Select the given module
    if (id == rawId) {
      // global displacement
      align::GlobalVector gvector(globalDisplacements.at(0), globalDisplacements.at(1), globalDisplacements.at(2));
      theTIBTIDAlignable->move(gvector);

      // global rotation
      if (toAndFro) {
        align::RotationType theResultRotation = backwardRotation * forwardRotation.transposed();
        theTIBTIDAlignable->rotateInGlobalFrame(theResultRotation);
      } else {
        theTIBTIDAlignable->rotateInGlobalFrame(backwardRotation);
      }
    }
  }
}

//__________________________________________________________________
//
void TrackerAlignment::saveToDB(void) {
  // Output POOL-ORA objects
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())  // Die if not available
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Retrieve and store
  Alignments* alignments = theAlignableTracker->alignments();
  AlignmentErrorsExtended* alignmentErrors = theAlignableTracker->alignmentErrors();

  //   if ( poolDbService->isNewTagRequest(theAlignRecordName) )
  //     poolDbService->createNewIOV<Alignments>( alignments, poolDbService->endOfTime(),
  //                                              theAlignRecordName );
  //   else
  //     poolDbService->appendSinceTime<Alignments>( alignments, poolDbService->currentTime(),
  //                                                 theAlignRecordName );
  poolDbService->writeOne<Alignments>(alignments, poolDbService->currentTime(), theAlignRecordName);
  //   if ( poolDbService->isNewTagRequest(theErrorRecordName) )
  //     poolDbService->createNewIOV<AlignmentErrorsExtended>( alignmentErrors,
  //                                                   poolDbService->endOfTime(),
  //                                                   theErrorRecordName );
  //   else
  //     poolDbService->appendSinceTime<AlignmentErrorsExtended>( alignmentErrors,
  //                                                      poolDbService->currentTime(),
  //                                                      theErrorRecordName );
  poolDbService->writeOne<AlignmentErrorsExtended>(alignmentErrors, poolDbService->currentTime(), theErrorRecordName);
}

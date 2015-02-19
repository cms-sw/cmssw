### Subdetector Enumerator Mapping

In the `TrackerGeometry` each `GeomDet` has a type from which it is possible to obtain the subdetector. The subdetector
is coded according to the `GeomDetEnumerators::SubDetector` enumerators in the Geometry/CommonDetUnit package. They contains
all the possible Tracker subdetectors for the present and upgraded detector. In the process of building the `TrackerGeometry`
object from the `GeometricDet` tree (created by the code in the Geometry/TrackerNumberingBuilder package) the
`GeometricDet::GDEnumType` enumerators are mapped onto the `GeomDetEnumerators::SubDetector` enumerators. In some cases
the Tracker subdetector `GeomDetEnumerators::SubDetector` enumerators have to be mapped to the first six enumerators
because they are used as index for vectors and arrays of six elements. This is achieved with the map 
`GeomDetEnumerators::subDetGeom[GeomDetEnumerators::SubDetector id]` which returns one of the first six enumerators.
To simplify the conditional statements based on the value of the `GeomDetEnumerators::SubDetector` enumerators, a few
helper methods are available (and other can be added if needed): `GeomDetEnumerators::isBarrel(subdet)`,
`GeomDetEnumerators::isEndcap(subdet)`, `GeomDetEnumerators::isTrackerPixel(subdet)`, 
`GeomDetEnumerators::isTrackerStrip(subdet)` and equivalent methods in the `GeomDetType` class. The present map between
enumerators and the returned values of the above methods are summarized in the table below:

| `GeometricDet::GDEnumType` | `GeomDetEnumerators::SubDetector` | `GeomDetEnumerators::subDetGeom[id]` | `isTrackerPixel` | `isTrackerStrip` | `isBarrel` | `isEndcap` | 
|-------|------|--------|------|------|-------|-------|
| `PixelBarrel` | `PixelBarrel` | `PixelBarrel` | `true` | `false` | `true` | `false` |
| `PixelEndCap` | `PixelEndcap` | `PixelEndcap` | `true` | `false` | `false` | `true` |
| `TIB` | `TIB` | `TIB` | `false` | `true` | `true` | `false` |
| `TID` | `TID` | `TID` | `false` | `true` | `false` | `true` |
| `TOB` | `TOB` | `TOB` | `false` | `true` | `true` | `false` |
| `TEC` | `TEC` | `TEC` | `false` | `true` | `false` | `true` |
| `PixelPhase1Barrel` | `P1PXB` | `PixelBarrel` | `true` | `false` | `true` | `false` |
| `PixelPhase1EndCap` | `P1PXEC` | `PixelEndcap` | `true` | `false` | `false` | `true` |
| `PixelPhase2EndCap` | `P2PXEC` | `PixelEndcap` | `true` | `false` | `false` | `true` |
| `OTPhase2Barrel` | `P2OTB` | `TOB` | `true` | `false` | `true` | `false` |
| `OTPhase2EndCap` | `P2OTEC` | `TID` | `true` | `false` | `false` | `true` |

### `TrackerGeometry` useful methods

Since the `GeomDet` are not always available, two help methods are available: `TrackerGeometry::geomDetSubDetector(int i)` 
which returns the value of the `GeomDetEnumerators::SubDetector` enumerator which correspond to the `DetId` subdetector `i`,
and `TrackerGeometry::numberOfLayers(int i)` which returns the number of layers of the `DetId` subdetector `i`. The values
of these methods for the three scenarios available so far are described in the tables below. In addition the method 
`TrackerGeometry::isThere(GeomDetEnumerators::SubDetector subdet)` can be used to test if the geometry contains the subdetector subdet.

* Present detector

| `DetId::subDetId()` | `TrackerGeometry::geomDetSubDetector(subdet)` | `TrackerGeometry::numberOfLayers(subdet)` |
|--------|--------|-------|
| 1 | `GeomDetEnumerators::PixelBarrel` | 3 |
| 2 | `GeomDetEnumerators::PixelEndcap` | 2 |
| 3 | `GeomDetEnumerators::TIB` | 4 |
| 4 | `GeomDetEnumerators::TID` | 3 |
| 5 | `GeomDetEnumerators::TOB` | 6 |
| 6 | `GeomDetEnumerators::TEC` | 9 |

* Phase1 Tracker

| `DetId::subDetId()` | `TrackerGeometry::geomDetSubDetector(subdet)` | `TrackerGeometry::numberOfLayers(subdet)` |
|--------|--------|-------|
| 1 | `GeomDetEnumerators::P1PXB` | 4 |
| 2 | `GeomDetEnumerators::P1PXEC` | 3 |
| 3 | `GeomDetEnumerators::TIB` | 4 |
| 4 | `GeomDetEnumerators::TID` | 3 |
| 5 | `GeomDetEnumerators::TOB` | 6 |
| 6 | `GeomDetEnumerators::TEC` | 9 |

* Phase2 Tracker
 
| `DetId::subDetId()` | `TrackerGeometry::geomDetSubDetector(subdet)` | `TrackerGeometry::numberOfLayers(subdet)` |
|--------|--------|-------|
| 1 | `GeomDetEnumerators::P1PXB` | 4 |
| 2 | `GeomDetEnumerators::P2PXEC` | 10 |
| 3 | `GeomDetEnumerators::invalidDet` | 0 |
| 4 | `GeomDetEnumerators::P2OTEC` | 5 |
| 5 | `GeomDetEnumerators::P2OTB` | 6 |
| 6 | `GeomDetEnumerators::invalidDet` | 0 |
 

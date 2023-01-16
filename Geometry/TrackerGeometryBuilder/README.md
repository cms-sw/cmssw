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

| `GeometricDet::GDEnumType` | `GeomDetEnumerators::SubDetector` | `GeomDetEnumerators::subDetGeom[id]` | `isTrackerPixel` | `isTrackerStrip` | `isInnerTracker` | `isOuterTracker` | `isBarrel` | `isEndcap` |
|-------|------|--------|------|------|-------|-------|-------|-------|
| `PixelBarrel` | `PixelBarrel` | `PixelBarrel` | `true` | `false` | `true` |  `false` | `true` | `false` |
| `PixelEndCap` | `PixelEndcap` | `PixelEndcap` | `true` | `false` | `true` |  `false` | `false` | `true` |
| `TIB` | `TIB` | `TIB` | `false` | `true` | `false` | `true` | `true` | `false` |
| `TID` | `TID` | `TID` | `false` | `true` | `false` | `true` | `false` | `true` |
| `TOB` | `TOB` | `TOB` | `false` | `true` | `false` | `true` |`true` | `false` |
| `TEC` | `TEC` | `TEC` | `false` | `true` | `false` | `true` |`false` | `true` |
| `PixelPhase1Barrel` | `P1PXB` | `PixelBarrel` | `true` | `false` | `true` |  `false` | `true` | `false` |
| `PixelPhase1EndCap` | `P1PXEC` | `PixelEndcap` | `true` | `false` | `true` |  `false` | `false` | `true` |
| `PixelPhase2Barrel` | `P2PXB` | `PixelBarrel` | `true` | `false` |`true` |  `false` | `true` | `false` |
| `PixelPhase2EndCap` | `P2PXEC` | `PixelEndcap` | `true` | `false` | `true` | `false` | `false` | `true` |
| `OTPhase2Barrel` | `P2OTB` | `TOB` | `true` | `false` | `false` | `true` |`true` | `false` |
| `OTPhase2EndCap` | `P2OTEC` | `TID` | `true` | `false` | `false` | `true` | `false` | `true` |

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
| 1 | `GeomDetEnumerators::P2PXB` | 4 |
| 2 | `GeomDetEnumerators::P2PXEC` | 12 |
| 3 | `GeomDetEnumerators::invalidDet` | 0 |
| 4 | `GeomDetEnumerators::P2OTEC` | 5 |
| 5 | `GeomDetEnumerators::P2OTB` | 6 |
| 6 | `GeomDetEnumerators::invalidDet` | 0 |
 
* ModuleTypes in  in `TrackerGeometry` class

The `TrackerGeometry` class updated to keep module type information with the highest `DetId` of that type
so that using `DetId` one can access the type. The `ModuleType` is contructed directly from the names defined in the
`Geometry` xml definitions 

Following types are used

| `TrackerGeometry::ModuleType` | `Description` |
|--------|-------|
| TrackerGeometry::UNKNOWN| Undefined            |                 
| TrackerGeometry::PXB    | Pixel Bar            |
| TrackerGeometry::PXF    | Pixel For            |
| TrackerGeometry::IB1    | IB1                  |
| TrackerGeometry::IB2    | IB2                  |
| TrackerGeometry::OB1    | OB1                  |
| TrackerGeometry::OB2    | OB2                  |
| TrackerGeometry::W1A    | W1A                  |
| TrackerGeometry::W2A    | W2A                  |
| TrackerGeometry::W3A    | W3A                  |
| TrackerGeometry::W1B    | W1B                  |
| TrackerGeometry::W2B    | W2B                  |
| TrackerGeometry::W3B    | W3B                  |
| TrackerGeometry::W4     | W4                   |
| TrackerGeometry::W5     | W5                   |
| TrackerGeometry::W6     | W6                   |
| TrackerGeometry::W7     | W7                   |
| TrackerGeometry::Ph1PXB | Phase 1 Pixel Barrel |
| TrackerGeometry::Ph1PXF | Phase 1 Pixel Endcap |
| TrackerGeometry::Ph2PXB | Phase 2 Pixel Barrel, planar sensor |
| TrackerGeometry::Ph2PXF | Phase 2 Pixel Barrel, planar sensor |
| TrackerGeometry::Ph2PXB3D | Phase 2 Pixel Barrel, 3D sensor |
| TrackerGeometry::Ph2PXF3D | Phase 2 Pixel Barrel, 3D sensor |
| TrackerGeometry::Ph2PSP | Phase 2 PS, p-sensor |
| TrackerGeometry::Ph2PSS | Phase 2 PS, s-sensor |
| TrackerGeometry::Ph2SS  | Phase2 2S            |

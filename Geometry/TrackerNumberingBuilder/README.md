In this package the `GeometricDet` tree of the `TrackerGeometry` object in the `TrackerDigiGeometryRecord` is created, the DetIds are assigned to the different components in the tree according to the schema which is defined in the xml description of trackerParameters.xml for the `TrackerGeometricDetESModule`, and the configuration of the `TrackerTopology` object is configured to be able to decode the DetIds using
the same schema.

The graph of the `GeometricDet` tree can be found in [this file](doc/GeometricDetBuilder.png)

## Available DetId Schemas
The predefined DetId schemas available in this package are:
* The Run 1 (aka _present_) detector DetId schema
* The Phase1 detector DetId schema where the pixel detector is replaced by the upgraded one
* The Phase 2 upgrade detectors DetId schema where the strip tracker is replaced by the upgraded outer tracker

In the table below the DetId levels which are in normal font represents _real_ hierarchy levels which are present 
also in the `GeometricDet` tree which is build in parallel to the DetId assignment. Those levels which are in _italic_ font are _fake_ levels and are not known by the GeometricDet tree.
When the name of the `TrackerTopology` method is written in _italic_, it means that its name does not reflect the actual attribute of the DetId which is returned. This is because, so far, the names of the methods are hardcoded and reflect the present Tracker detector. In addition two generic methods, `TrackerTopology::layer(id)` and `TrackerTopology::side(id)` can be used to determine the layer/disk number and the side of the endcap subdetectors.

### Run 1 Detector DetId schema
The Run 1 detector DetId schema profits of all the six available subdetectors (from 1 to 6) and it is defined as follows

* Subdetector 1 (`DetId::subDetId() == PixelSubdetector::PixelBarrel`): Pixel Barrel

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|-----|----|-----|
| _not used_ | 20 | 0x1F | 5 | | |
| Layer | 16 | 0xF | 4 | pxbLayer(id) or layer(id) | increasing r |
| Ladder | 8 | 0xFF | 8 | pxbLadder(id) | increasing phi |
| Module | 2 | 0x3F | 6 | pxbModule(id) | increasing z |
| _not used_ | 0 | 0x3 | 2 | | |

* Subdetector 2 (`DetId::subDetId() == PixelSubdetector::PixelEndcap`): Pixel Forward

| Name | start bit | hex mask | bit size | `TrackerToplogy` method | Notes |
|------|-----------|-----------|----|-----|-----|
| subdetector part | 23 | 0x3 | 2 | pxfSide(id) or side(id) | 1=FPIX- 2=FPIX+ |
| _not used_ | 20 | 0x7 | 3 | | |
| Disk | 16 | 0xF | 4 | pxfDisk(id) or layer(id) | increasing abs(z) |
| _Blade_ | 10 | 0x3F | 6 | pxfBlade(id) | increasing phi |
| Panel | 8 | 0x3 | 2 | pxfPanel(id) | 1=forward 2=backward |
| Module | 2 | 0x3F | 6 | pxfModule(id) | increasing r |
| _not used_ | 0 | 0x3 | 2 | | |

* Subdetector 3 (`DetId::subDetId() == StripSubdetector::TIB`): TIB

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|-----|----|-----|
| not used | 17 | 0xFF | 8 | | |
| Layer | 14 |0x7 | 3 | tibLayer(id) or layer(id) | increasing r |
| _subdetector part_ | 12 | 0x3 | 2 | tibSide(id) | 1=TIB- 2=TIB+ |
| _Layer side_ | 10 | 0x3 | 2 | tibOrder(id) | 1=internal 2=external |
| String | 4 | 0x3F | 6 | tibString(id) | increasing phi |
| Module | 2 | 0x3 | 2 | tibModule(id) | increasing abs(z) |
| Module type | 0 | 0x3 | 2 | tibStereo(id) or tibGlued(id) | 1=stereo, 2=rphi, 0=pair |

* Subdetector 4 (`DetId::subDetId() == StripSubdetector::TID`): TID

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|-----|----|-----|
| _not used_ | 15 | 0x3FF | 10 | | |
| subdetector part | 13 |0x3 | 2 | tidSide(id) or side(id) | 1=TID- 2=TID+ |
| Disk | 11 | 0x3 | 2 | tidWheel(id) or layer(id) | increasing abs(z) |
| Ring | 9 | 0x3 | 2 | tidRing(id) | increasing r |
| _Disk side_ | 7 | 0x3 | 2 | tidOrder(id) | 1=back 2=front |
| Module | 2 | 0x1F | 5 | tidModule(id) | increasing phi |
| Module type | 0 | 0x3 | 2 | tidStereo(id) or tidGlued(id) | 1=stereo, 2=rphi, 0=pair |

* Subdetector 5 (`DetId::subDetId() == StripSubdetector::TOB`): TOB 

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|-----|----|----|
| _not used_ | 17 | 0xFF | 8 | | |
| Layer | 14 |0x7 | 3 | tobLayer(id) or layer(id) | increasing r |
| _subdetector part_ | 12 | 0x3 | 2 | tobSide(id) | 1=TIB- 2=TIB+ |
| Rod | 5 | 0x7F | 7 | tobRod(id) | increasing phi |
| Module | 2 | 0x7 | 3 | tobModule(id) | increasing abs(z) |
| Module type | 0 | 0x3 | 2 | tobStereo(id) or tobGlued(id) | 1=stereo, 2=rphi, 0=pair |

* Subdetector 6 (`DetId::subDetId() == StripSubdetector::TEC`): TEC

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|-----|----|----|
| _not used_ | 20 | 0x3F | 5 | | |
| subdetector part | 18 |0x3 | 2 | tecSide(id) or side(id) | 1=TEC- 2=TEC+ |
| Wheel | 14 | 0xF | 4 | tecWheel(id) or layer(id) | increasing abs(z) |
| _Wheel side_ | 12 | 0x3 | 2 | tecOrder(id) | 1=back 2=front |
| Petal | 8 | 0xF | 4 | tecPetal(id) | increasing phi |
| Ring | 5 | 0x7 | 3 | tecRing(id) | increasing r |
| Module | 2 | 0x7 | 3 | tecModule(id) | increasing phi |
| Module type | 0 | 0x3 | 2 | tecStereo(id) or tecGlued(id) | 1=stereo, 2=rphi, 0=pair |

A more detailed description of the SiStrip Tracker DetId schema can be found in this CMS Internal Note: [http://cms.cern.ch/iCMS/jsp/openfile.jsp?type=IN&year=2007&files=IN2007_020.pdf]

The configuration names for this detid schema are `trackerNumberingGeometry_cfi` (to run on geometry built from xml files) or `trackerNumberingGeometryDB_cfi` (to run on geometry from DB) for `TrackerGeometricDetESModule` and `trackerTopologyConstants_cfi` for `TrackerTopology`
The xml description of tracker parameters for this detid schema is in [Geometry/TrackerCommonData/data/trackerParameters.xml](../TrackerCommonData/data/trackerParameters.xml)

### Phase 1 Upgrade Detector DetId schema
The phase 1 detector DetId schema differs from that of the Run 1 detector only in the first two subdetectors which
corresponds to the Pixel Barrel and Forward detector. Therefore only them will be repeated here:

* Subdetector 1 (`DetId::subDetId() == PixelSubdetector::PixelBarrel`): Phase1 Pixel Barrel

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|-----|----|-----|
| _not used_ | 24 | 0x1 | 1 | | |
| Layer | 20 | 0xF | 4 | pxbLayer(id) or layer(id) | increasing r |
| Ladder | 12 | 0xFF | 8 | pxbLadder(id) | increasing phi |
| Module | 2 | 0x3FF | 10 | pxbModule(id) | increasing z |
| _not used_ | 0 | 0x3 | 2 | | |

* Subdetector 2 (`DetId::subDetId() == PixelSubdetector::PixelEndcap`): Phase1 Pixel Forward

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|----|-----|-----|
| subdetector part | 23 | 0x3 | 2 | pxfSide(id) or side(id) | 1=FPIX- 2=FPIX+ |
| _not used_ | 22 | 0x1 | 1 | | |
| Disk | 18 | 0xF | 4 | pxfDisk(id) or layer(id) | increasing abs(z) |
| _Blade_ | 12 | 0x3F | 6 | pxfBlade(id) | increasing phi and r: first inner ring blades and the outer ring blades |
| Panel | 10 | 0x3 | 2 | pxbPanel(id) | 1=forward 2=backward |
| Module | 2 | 0xFF | 8 | _pxbModule(id)_ | always = 1 |
| _not used_ | 0 | 0x3 | 2 | | |

Subdetectors 3 to 6 are as for the Run 1 detector since the SiStrip Tracker is the same in phase1.

The configuration names for this detid schema are `trackerNumberingGeometry_cfi` (to run on geometry built from xml files) or `trackerNumberingGeometryDB_cfi` (to run on geometry from DB) for `TrackerGeometricDetESModule` and `trackerTopology2017Constants_cfi` for `TrackerTopology`
The xml description of tracker parameters for this detid schema is in [Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml](../TrackerCommonData/data/PhaseI/trackerParameters.xml)

### Phase 2 Upgrade Detector DetId schema
The phase 2 detector DetId schema is identical to the one of the phase 1 detector for the inner pixel detector while for the outer tracker subdetector 5, for the barrel, and subdetector 4, for the endcap, are used. In some cases the name of the `TrackerTopology` methods is not so meaningful.
 
* Subdetector 1: (`DetId::subDetId() == PixelSubdetector::PixelBarrel`): Phase1 Pixel Barrel

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|-----|----|-----|
| _not used_ | 24 | 0x1 | 1 | | |
| Layer | 20 | 0xF | 4 | pxbLayer(id) or layer(id) | increasing r |
| Ladder | 12 | 0xFF | 8 | pxbLadder(id) | increasing phi |
| Module | 2 | 0x3FF | 10 | pxbModule(id) | increasing z |
| _not used_ | 0 | 0x3 | 2 | | |

* Subdetector 2: (`DetId::subDetId() == PixelSubdetector::PixelEndcap`): Phase2 Pixel Forward

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|----|-----|-----|
| subdetector part | 23 | 0x3 | 2 | pxfSide(id) or side(id) | 1=FPIX- 2=FPIX+ |
| _not used_ | 22 | 0x1 | 1 | | |
| Disk | 18 | 0xF | 4 | pxfDisk(id) or layer(id) | increasing abs(z) |
| _Blade_ | 12 | 0x3F | 6 | pxfBlade(id) | increasing phi and r: first inner ring blades and the outer ring blades |
| Panel | 10 | 0x3 | 2 | pxbPanel(id) | 1=forward 2=backward |
| Module | 2 | 0xFF | 8 | _pxbModule(id)_ | always = 1 |
| _not used_ | 0 | 0x3 | 2 | | |

* Subdetector 5  (`DetId::subDetId() == StripSubdetector::TOB`): Phase2 Outer Tracker Barrel

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|-----|----|-----|
| _not used_ | 24 | 0x1 | 1 | | |
| Layer | 20 | 0xF | 4 | tobLayer(id) or layer(id) | increasing r |
| Ladder | 12 | 0xFF | 8 | tobRod(id) | increasing phi |
| Module | 2 | 0x3FF | 10 | tobModule(id) | increasing z and in the same pt module modules are sorted by increasing r |
| _not used_ | 0 | 0x3 | 2 | | |

* Subdetector 4  (`DetId::subDetId() == StripSubdetector::TID`): Phase2 Outer Tracker Endcap

| Name | start bit | hex mask | bit size | `TrackerTopology` method | Notes |
|------|-----------|-----------|----|-----|----|
| subdetector part | 23 | 0x3 | 2 | tidSide(id) or side(id) | 1=-ve 2=+ve |
| _not used_ | 22 | 0x1 | 1 | | |
| Disk | 18 | 0xF | 4 | tidDisk(id) or side(id) | increasing abs(z) |
| _Ring_ | 12 | 0x3F | 6 | tidRing(id) | increasing r |
| Panel | 10 | 0x3 | 2 | _tidOrder(id)_ | always = 1 |
| Module | 2 | 0xFF | 8 | tidModule(id) | increasing phi and modules in the same pt module are sorted by increasing abs(z) |
| _not used_ | 0 | 0x3 | 2 | | |

The configuration names for this detid schema are `trackerNumberingGeometry_cfi` (to run on geometry built from xml files) or `trackerNumberingGeometryDB_cfi` (to run on geometry from DB) for `TrackerGeometricDetESModule` and `trackerTopology2023Constants_cfi` for `TrackerTopology`
The xml description of tracker parameters for this detid schema is in [Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml](../TrackerCommonData/data/PhaseII/trackerParameters.xml)

### Subdetector `GeometricDet` Enumerators

The link between the subdetectors described in the geometry and the `DetId::subDetId()` is created by the `GeometricDet::GDEnumType` enumerators. Each subdetector name in the Tracker DDD is associated to a `GeometricDet::GDEnumType` enumerator which has to be of the form n*100+m where m is between 1 and 6 and it will correspond to the `DetId::subDetId()` value. The present link table is:

| Tk DDD name | `GeometricDet::GDEnumType` | `DetId::subDetId()` |
|-------------|------------------------|---------------------|
| PixelBarrel | `PixelBarrel`=1 | 1=`PixelSubdetector::PixelBarrel` |
| PixelEndcapSubDet | `PixelEndCap`=2 | 2=`PixelSubdetector::PixelEndcap` |
| TIB | `TIB`=3 | 3=`StripSubdetector::TIB` |
| TID | `TID`=4 | 4=`StripSubdetector::TID` |
| TOB | `TOB`=5 | 5=`StripSubdetector::TOB` |
| TEC | `TEC`=6 | 6=`StripSubdetector::TEC` |
| PixelPhase1Barrel | `PixelPhase1Barrel`=101 | 1=`PixelSubdetector::PixelBarrel` |
| PixelPhase1EndcapSubDet | `PixelPhase1EndCap`=102 | 2=`PixelSubdetector::PixelEndcap` |
| PixelPhase2EndcapSubDet | `PixelPhase2EndCap`=202 | 2=`PixelSubdetector::PixelEndcap` |
| Phase2OTBarrel | `OTPhase2Barrel`=205 | 5=`StripSubdetector::TOB` |
| Phase2OTEndcap | `OTPhase2EndCap`=204 | 4=`StripSubdetector::TID` |

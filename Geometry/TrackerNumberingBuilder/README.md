In this package the `GeometricDet` tree of the `TrackerGeometry` object in the `TrackerDigiGeometryRecord` is created, the DetIds are assigned to the different components in the tree according to the schema which is defined in the configuration
of `TrackerGeometricDetESModule`, and the `TrackerTopology` object is configured to be able to decode the DetIds using
the same schema.

The graph of the `GeometricDet` tree can be found in [this file](GeometricDetBuilder.png)

## Available DetId Schemas
The predefined DetId schemas available in this package are:
* The Run 1 (aka _present_) detector DetId schema
* The phase 1 and phase 2 upgrade detectors DetId schema. It is common to the two detectors. See below for the details.

In the table below the DetId levels which are in normal font represents _real_ hierarchy levels which are present 
also in the `GeometricDet` tree which is build in parallel to the DetId assignment. Those levels which are in _italic_ font are _fake_ levels and are not known by the GeometricDet tree.

### Run 1 Detector DetId schema
The Run 1 detector DetId schema profits of all the six available subdetectors (from 1 to 6) and it is defined as follows

* Subdetector 1: Barrel Pixel (`GeometricDet::PixelBarrel`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|-----|----|
| _not used_ | 20 | 0x1F | 5 | |
| Layer | 16 | 0xF | 4 | increasing r |
| Ladder | 8 | 0xFF | 8 | increasing phi |
| Module | 2 | 0x3F | 6 | increasing z |
| _not used_ | 0 | 0x3 | 2 | |

* Subdetector 2: Forward Pixel (`GeometricDet::PixelEndCap`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|----|-----|
| subdetector part | 23 | 0x3 | 2 | 1=FPIX- 2=FPIX+ |
| _not used_ | 20 | 0x7 | 3 | |
| Disk | 16 | 0xF | 4 | increasing abs(z) |
| _Blade_ | 10 | 0x3F | 6 | increasing phi |
| Panel | 8 | 0x3 | 2 | 1=forward 2=backward |
| Module | 2 | 0x3F | 6 | increasing r |
| _not used_ | 0 | 0x3 | 2 | |

* Subdetector 3: TIB (`GeometricDet::TIB`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|-----|----|
| not used | 17 | 0xFF | 8 | |
| Layer | 14 |0x7 | 3 | increasing r |
| _subdetector part_ | 12 | 0x3 | 2 | 1=TIB- 2=TIB+ |
| _Layer side_ | 10 | 0x3 | 2 | 1=internal 2=external |
| String | 4 | 0x3F | 6 | increasing phi |
| Module | 2 | 0x3 | 2 | increasing abs(z) |
| Module type | 0 | 0x3 | 2 | 1=stereo, 2=rphi, 0=pair |

* Subdetector 4: TID (`GeometricDet::TID`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|-----|----|
| _not used_ | 15 | 0x3FF | 10 | |
| subdetector part | 13 |0x3 | 2 | 1=TID- 2=TID+ |
| Disk | 11 | 0x3 | 2 | increasing abs(z) |
| Ring | 9 | 0x3 | 2 | increasing r |
| _Disk side_ | 7 | 0x3 | 2 | 1=back 2=front |
| Module | 2 | 0x1F | 5 | increasing phi |
| Module type | 0 | 0x3 | 2 | 1=stereo, 2=rphi, 0=pair |

* Subdetector 5: TOB (`GeometricDet::TOB`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|-----|----|
| _not used_ | 17 | 0xFF | 8 | |
| Layer | 14 |0x7 | 3 | increasing r |
| _subdetector part_ | 12 | 0x3 | 2 | 1=TIB- 2=TIB+ |
| Rod | 5 | 0x7F | 7 | increasing phi |
| Module | 2 | 0x7 | 3 | increasing abs(z) |
| Module type | 0 | 0x3 | 2 | 1=stereo, 2=rphi, 0=pair |

* Subdetector 6: TEC (`GeometricDet::TEC`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|-----|----|
| _not used_ | 20 | 0x3F | 5 | |
| subdetector part | 18 |0x3 | 2 | 1=TEC- 2=TEC+ |
| Wheel | 14 | 0xF | 4 | increasing abs(z) |
| _Wheel side_ | 12 | 0x3 | 2 | 1=back 2=front |
| Petal | 8 | 0xF | 4 | increasing phi |
| Ring | 5 | 0x7 | 3 | increasing r |
| Module | 2 | 0x7 | 3 | increasing phi |
| Module type | 0 | 0x3 | 2 | 1=stereo, 2=rphi, 0=pair |

A more detailed description of the SiStrip Tracker DetId schema can be found in this CMS Internal Note: [http://cms.cern.ch/iCMS/jsp/openfile.jsp?type=IN&year=2007&files=IN2007_020.pdf]

The configuration names for this detid schema are `trackerNumberingGeometry_cfi` for `TrackerGeometricDetESModule` and `trackerTopologyConstants_cfi` for `TrackerTopology`

### Phase 1 Upgrade Detector DetId schema
The phase 1 detector DetId schema differs from that of the Run 1 detector only in the first two subdetectors which
corresponds to the Pixel Barrel and Forward detector. Therefore only them will be repeated here:

* Subdetector 1: Barrel Pixel (`GeometricDet::PixelBarrel`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|-----|----|
| _not used_ | 24 | 0x1 | 1 | |
| Layer | 20 | 0xF | 4 | increasing r |
| Ladder | 12 | 0xFF | 8 | increasing phi |
| Module | 2 | 0x3FF | 10 | increasing z |
| _not used_ | 0 | 0x3 | 2 | |

* Subdetector 2: Forward Pixel (`GeometricDet::PixelPhase1EndCap`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|----|-----|
| subdetector part | 23 | 0x3 | 2 | 1=FPIX- 2=FPIX+ |
| _not used_ | 22 | 0x1 | 1 | |
| Disk | 18 | 0xF | 4 | increasing abs(z) |
| _Blade_ | 12 | 0x3F | 6 | increasing phi and r: first inner ring blades and the outer ring blades |
| Panel | 10 | 0x3 | 2 | 1=forward 2=backward |
| Module | 2 | 0xFF | 8 | always = 1 |
| _not used_ | 0 | 0x3 | 2 | |

Subdetectors 3 to 6 are as for the Run 1 detector since the SiStrip Tracker is the same in phase1.

The configuration names for this detid schema are `trackerNumberingSLHCGeometry_cfi` for `TrackerGeometricDetESModule` and `trackerTopologySLHCConstants_cfi` for `TrackerTopology`

### Phase 2 Upgrade Detector DetId schema
The phase 2 detector DetId schema is formally identical to the one of the phase 1 detector but only subdetectors 1 and 2
are used. Furthermore the meaning of some levels is different if they apply to the inner pixel detector or to the 
phase 2 OT detector 

* Subdetector 1: Barrel (inner pixel + outer tracker) (`GeometricDet::PixelBarrel`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|-----|----|
| _not used_ | 24 | 0x1 | 1 | |
| Layer | 20 | 0xF | 4 | increasing r |
| Ladder | 12 | 0xFF | 8 | increasing phi |
| Module | 2 | 0x3FF | 10 | increasing z. In the outer tracker modules in the same pt module are sorted by increasing r |
| _not used_ | 0 | 0x3 | 2 | |

* Subdetector 2: Forward (inner pixel + outer tracker) (`GeometricDet::PixelPhase2EndCap`)

| Name | start bit | hex mask | bit size | Notes |
|------|-----------|-----------|----|-----|
| subdetector part | 23 | 0x3 | 2 | 1=FPIX- 2=FPIX+ |
| _not used_ | 22 | 0x1 | 1 | |
| Disk | 18 | 0xF | 4 | inner pixel before outer tracker, then increasing abs(z) |
| _Blade_ | 12 | 0x3F | 6 | inner pixel: increasing phi and r: first inner ring blades and the outer ring blades; outer tracker: "blade" is used for the ring (increasing r) |
| Panel | 10 | 0x3 | 2 | inner pixel: 1=forward 2=backward; outer tracker: always = 1 |
| Module | 2 | 0xFF | 8 | inner pixel: always = 1; outer tracker: increasing phi and modules in the same pt module are sorted by increasing abs(z) |
| _not used_ | 0 | 0x3 | 2 | |

The configuration names for this detid schema are `trackerNumberingSLHCGeometry_cfi` for `TrackerGeometricDetESModule` and `trackerTopologySLHCConstants_cfi` for `TrackerTopology`

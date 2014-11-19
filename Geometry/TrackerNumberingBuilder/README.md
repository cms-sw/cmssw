In this package the GeometricDet tree of the TrackerGeometry object in the TrackerDigiGeometryRecord is created, the 
DetIds are assigned to the different components in the tree according to the schema which is defined in the configuration
of TrackerGeometricDetESModule, and the TrackerTopology object is configured to be able to decode the DetIds using
the same schema.

# Available DetId Schemas
The predefined DetId schemas available in this package are:
* The Run 1 (aka _present_) detector DetId schema
* The phase 1 and phase 2 upgrade detectors DetId schema. It is common to the two detectors. See below for the details
In the table below the DetId levels which are highlighted in bold represents _real_ hierarchy levels which are present 
also in the GeometricDet tree which is build in parallel to the DetId assignment. Those levels which are in normal font 
are _fake_ levels and are not known by the GeometricDet tree.

## Run 1 Detector DetId schema
The Run 1 detector DetId schema profits of all the six available subdetectors (from 1 to 6) and it is defined as follows
* Subdetector 1 : Barrel Pixel

| Name | start bit | bit range |
|------|-----------|-----------|
| Layer | xx | yy |
|-------|----|----|

* Subdetector 2: Forward Pixel

| Name | start bit | bit range |
|------|-----------|-----------|
| nnnn | xx | yy |
|------|----|----|

A more detailed description of the SiStrip Tracker DetId schema can be found in this CMS Note

The configuration names for this detid schema are named xxxx for TrackerGeometricDetESModule and yyyy for 
TrackerTopology

## Upgrade Detector DetId schema
The phase 1 detector DetId schema differs from that of the Run 1 detector only in the first two subdetectors which
corresponds to the Pixel Barrel and Forward detector. Therefore only them will be repeated here:
* Subdetector 1 : Barrel Pixel

| Name | start bit | bit range |
|------|-----------|-----------|
| Layer | xx | yy |
|-------|----|----|

* Subdetector 2: Forward Pixel

| Name | start bit | bit range |
|------|-----------|-----------|
| nnnn | xx | yy |
|------|----|----|

The phase 2 detector DetId schema is formally identical to the one of the phase 1 detector but only subdetectors 1 and 2
are used. Furthermore the meaning of some levels is different if they apply to the inner pixel detector or to the 
phase 2 OT detector 

The configuration names for this detid schema are named xxxx for TrackerGeometricDetESModule and yyyy for 
TrackerTopology

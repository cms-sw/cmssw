Trigger geometries provide the following interfaces:
* Mapping between HGCAL sensor cells, trigger cells and motherboards
* Navigation between trigger cells

The available HGCAL trigger geometries are the following:
* `HGCalTriggerGeometryV9Imp2` (DEFAULT for geometries >= V9)
  - Implementation without trigger cell external mappings. Makes use of the `HGCSiliconDetId`, `HGCScintillatorDetId`, and `HGCalTriggerDetId`
  - Compatible with the HGCAL geometries >= V9
  - The trigger cell neighbors are not defined (no navigation)
* `HGCalTriggerGeometryV9Imp1` 
  - Similar implementation as `HGCalTriggerGeometryHexLayerBasedImp1`, but for the V9 geometry
  - Compatible with the V9 HGCAL geometry
* `HGCalTriggerGeometryHexLayerBasedImp1` (DEFAULT for V8 geometry)
  - The trigger cell mapping is defined over a full layer and is not constrained by wafer boundaries
  - Compatible with the V8 HGCAL geometry

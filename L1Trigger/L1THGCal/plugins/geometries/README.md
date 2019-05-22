Trigger geometries provide the following interfaces:
* Mapping between HGCAL sensor cells, trigger cells and motherboards
* Navigation between trigger cells

The available HGCAL trigger geometries are the following:
* `HGCalTriggerGeometryHexLayerBasedImp1` (DEFAULT)
  - The trigger cell mapping is defined over a full layer and is not constrained by wafer boundaries
  - Compatible with the V8 HGCAL geometry
* `HGCalTriggerGeometryV9Imp2`
  - Implementation without trigger cell external mappings. Makes use of the `HGCSiliconDetId`, `HGCScintillatorDetId`, and `HGCalTriggerDetId`
  - Compatible with the V9 HGCAL geometry
  - The trigger cell neighbors are not defined (no navigation)
* `HGCalTriggerGeometryV9Imp1`
  - Similar implementation as `HGCalTriggerGeometryHexLayerBasedImp1`, but for the V9 geometry
  - Compatible with the V9 HGCAL geometry
* `HGCalTriggerGeometryHexImp2` (DEPRECATED)
  - The trigger cell mapping is defined within single wafers. Trigger cells are therefore constrained by the wafer boundaries
  - The trigger cells in the BH section are not defined
* `HGCalTriggerGeometryHexImp1` (DEPRECATED)
  - The trigger cell mapping is defined over the full detector
  - The trigger cells in the BH section are not defined
  - The trigger cell neighbors are not defined (no navigation)
* `HGCalTriggerGeometryImp1` (DEPRECATED)
  - It is based on the old HGCAL square geometry
  - The trigger cell mapping is defined over the full detector
  - The trigger cell neighbors are not defined (no navigation)

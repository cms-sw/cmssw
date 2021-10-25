Trigger geometries provide the following interfaces:
* Mapping between HGCAL sensor cells, trigger cells, modules, lpGBTs and backend FPGAs

The available HGCAL trigger geometries are the following:
* `HGCalTriggerGeometryV9Imp3`
  - Compatible with the HGCAL geometries >= V9
  - All links mapping are available (elinks, lpGBT, BE links)
  - Backend FPGA mappings are available
  - Links and FPGA mappings are defined in external JSON files
* `HGCalTriggerGeometryV9Imp2` (DEFAULT)
  - Compatible with the HGCAL geometries >= V9
  - No links mapping. Only the number of elinks per module/ECON-T is available
  - Backend FPGA mappings are not available

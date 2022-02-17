Trigger geometries provide the following interfaces:
* Mapping between HGCAL sensor cells, trigger cells, modules, lpGBTs and backend FPGAs

The available HGCAL trigger geometries are the following:
* `HGCalTriggerGeometryV9Imp3` (DEFAULT)
  - Compatible with the HGCAL geometries >= V9
  - All links mapping are available (elinks, lpGBT, BE links)
  - Backend FPGA mappings are available
  - Links and FPGA mappings are defined in external JSON files
  - Mapping configs are available for 72 and 120 input links per Stage 1 FPGA.
  - These mappings correspond to a PU-driven distribution of elinks; there is no configuration corresponding to signal-driven elink distribution at the moment.
* `HGCalTriggerGeometryV9Imp2`
  - Compatible with the HGCAL geometries >= V9
  - No links mapping. Only the number of elinks per module/ECON-T is available. Both PU-driven and signal-driven elink distributions can be used.
  - Backend FPGA mappings are not available

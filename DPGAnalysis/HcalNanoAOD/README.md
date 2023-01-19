### HcalNanoAOD

This package provides modules for saving HCAL raw data to NanoAOD. Specifically, modules are provided for HB/HE/HF/HO digis, RecHits, trigger primitives (TPs), HF pre-RecHits, and calibration metadata. Also see DPGAnalysis/CaloNanoAOD for modules related to HCAL+particle flow. 

The digis are saved as a dense array (counting on compression to minimize the space consumed by 0s). The outputs are also sorted by DetId; the sorting is performed by the classes named *SortedTable.

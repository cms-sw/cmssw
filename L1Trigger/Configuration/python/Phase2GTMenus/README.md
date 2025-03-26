# L1 Phase-2 Global Trigger (P2GT) menus

Menus for the P2GT emulator (and firmware) are stored in this folder.

See the Menu twiki page for a general overview: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PhaseIIL1TriggerMenuTools

See the `README` in the P2GT package for details on how to write the menu configs:
[`L1Trigger/Phase2L1GT/README.md`](L1Trigger/Phase2L1GT/README.md)

Available menus:
* `step1_2023` –> first menu of 2023
* `step1_2024` -> the 2024 version 
* `step1_2024_explicitSeeds` –> the 2024 version with explicit definitions of individual objects

These menus can be specified via the cmsDriver configuration as `-s L1P2GT:step1_2024`.

Note that the full menu expanded configuration can be inspected using `edmConfigDump` on a `cmsRun` config that executes the `L1P2GT` step:
1. Produce the config via 
```
cmsDriver.py -s L1P2GT --conditions auto:phase2_realistic_T33 --filein file:file.root --no_exec --python_filename l1pgt_cfg.py
```
2. Dump the config via 
```
edmConfigDump l1pgt_cfg.py > edm_l1pgt_cfg.py
```
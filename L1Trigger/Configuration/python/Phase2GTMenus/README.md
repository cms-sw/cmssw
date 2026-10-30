# L1 Phase-2 Global Trigger (P2GT) menus

Menus for the P2GT emulator (and firmware) are stored in this folder.

See the Menu twiki page for a general overview: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PhaseIIL1TriggerMenuTools

See the `README` in the P2GT package for details on how to write the menu configs:
[`L1Trigger/Phase2L1GT/README.md`](L1Trigger/Phase2L1GT/README.md)

Available menus:
* `step1_2023` –> first menu of 2023
* `step1_2024` -> the 2024 version 
* `step1_2024_explicitSeeds` –> the 2024 version with explicit definitions of individual objects
* `step1and2_2026` -> `step1_2024` plus the step 2 seeds that P2GT can run
* `step1p5_2026` -> the step 1.5 menu, based on the Run 3 menu (2025 `v1_3_0`)
* `v0p0_2026` -> `step1p5_2026` plus the step 2 and extended-eta seeds that P2GT can run

The 2026 menus follow the Phase-2 L1 MenuTools menus of the same names (`configs/V50nano/rate_table/` in [Phase2-L1MenuTools](https://github.com/cms-l1-dpg/Phase2-L1MenuTools)), with path names `pX` for the MenuTools seed `L1_X`.
`step1_2024` remains the default.

These menus can be specified via the cmsDriver configuration as `-s L1P2GT:step1_2024`.

Note that the full menu expanded configuration can be inspected using `edmConfigDump` on a `cmsRun` config that executes the `L1P2GT` step:
1. Produce the config via 
```
cmsDriver.py -s L1P2GT --conditions auto:phase2_realistic_T35 --filein file:file.root --no_exec --python_filename l1pgt_cfg.py
```
2. Dump the config via 
```
edmConfigDump l1pgt_cfg.py > edm_l1pgt_cfg.py
```
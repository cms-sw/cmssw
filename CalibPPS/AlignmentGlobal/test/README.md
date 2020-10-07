# AlignmentGlobal/test

## Files
### Reference dataset
 - `config_reference_cff.py` - configuration (Event Setup) for the reference dataset
 - `input_files_reference_cff.py` - file with vstring of ROOT input files for the reference dataset 
 - `run_distributions_reference_cfg.py` - process configuration for PPSAlignmentWorker. Produces standard ROOT file with reference histograms for x alignment of the test dataset.
### Test dataset
 - `config_cff.py` - configuration (Event Setup) for the test dataset
 - `input_files_cff.py` - file with vstring of ROOT input files for the test dataset
 - `run_distributions_cfg.py` - process configuration for PPSAlignmentWorker. Produces DQMIO ROOT file with histograms for the harvester.
 - `run_analysis_manual_cfg.py` - process configuration for PPSAlignmentHarvester. Produces alignment results.

## Running instructions
```
cmsRun run_distributions_reference_cfg.py
cmsRun run_distributions_cfg.py
cmsRun run_analysis_manual_cfg.py
```

## Expected results
### x_alignment
 - RP 3: x = -3690.0 +- 17.9 um
 - RP 23: x = -41690.0 +- 17.2 um
 - RP 103: x = -2700.0 +- 16.9 um
 - RP 123: x = -41830.0 +- 16.0 um

### x_alignment_relative:
 - RP 3: x = 18985.6 +- 1.0 um
 - RP 23: x = -18985.6 +- 1.0 um
 - RP 103: x = 19484.1 +- 1.2 um
 - RP 123: x = -19484.1 +- 1.2 um

### x_alignment_relative_sl_fix:
 - RP 3: x = 18983.7 +- 0.2 um
 - RP 23: x = -18983.7 +- 0.2 um
 - RP 103: x = 19486.6 +- 0.3 um
 - RP 123: x = -19486.6 +- 0.3 um

### y_alignment:
 - RP 3: y = 3468.8 +- 44.1 um
 - RP 23: y = 4097.6 +- 44.8 um
 - RP 103: y = 3025.5 +- 77.8 um
 - RP 123: y = 3344.0 +- 66.1 um

### y_alignment_sl_fix:
 - RP 3: y = 3491.7 +- 10.7 um
 - RP 23: y = 4167.4 +- 11.6 um
 - RP 103: y = 2753.5 +- 18.2 um
 - RP 123: y = 3390.2 +- 17.3 um
# AlignmentGlobal/test

## Files
### Reference dataset
 - `config_reference_cff.py` - configuration (Event Setup) for the reference dataset
 - `input_files_reference_cff.py` - file with vstring of ROOT input files for the reference dataset 
 - `run_distributions_reference_cfg.py` - process configuration for `PPSAlignmentWorker`. Produces a standard ROOT file with reference histograms for the horizontal alignment.
### Test dataset
 - `config_cff.py` - configuration (Event Setup) for the test dataset
 - `input_files_cff.py` - file with vstring of ROOT input files for the test dataset
 - `run_distributions_cfg.py` - process configuration for `PPSAlignmentWorker`. Produces a DQMIO ROOT file with histograms for the harvester.
 - `run_analysis_cfg.py` - process configuration for `PPSAlignmentHarvester`. Runs the horizontal, the horizontal relative and the vertical alignment. Produces a text file with results and two ROOT files - one with DQM plots, and the other one with debug plots. Can be easily configured to produce an SQLite file with the results as well - check out the variables at the top of the file.

## Running instructions
```
cmsRun run_distributions_reference_cfg.py && cmsRun run_distributions_cfg.py && cmsRun run_analysis_cfg.py
```

## Expected results (alignment_results.txt)
```
1: x_alignment:
RP 3: shift (um) x = -3670.0 +- 19.8, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 23: shift (um) x = -41710.0 +- 20.6, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 103: shift (um) x = -2700.0 +- 18.8, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 123: shift (um) x = -41840.0 +- 19.5, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0


2: x_alignment_relative:
RP 3: shift (um) x = 18984.6 +- 1.2, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 23: shift (um) x = -18984.6 +- 1.2, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 103: shift (um) x = 19485.3 +- 1.5, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 123: shift (um) x = -19485.3 +- 1.5, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0

2: x_alignment_relative_sl_fix:
RP 3: shift (um) x = 18983.8 +- 0.3, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 23: shift (um) x = -18983.8 +- 0.3, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 103: shift (um) x = 19486.8 +- 0.3, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 123: shift (um) x = -19486.8 +- 0.3, y = 0.0 +- 0.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0


3: y_alignment:
RP 3: shift (um) x = 0.0 +- 0.0, y = 3443.3 +- 101.7, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 23: shift (um) x = 0.0 +- 0.0, y = 4190.3 +- 87.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 103: shift (um) x = 0.0 +- 0.0, y = 2885.9 +- 66.2, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 123: shift (um) x = 0.0 +- 0.0, y = 3491.8 +- 129.5, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0

3: y_alignment_sl_fix:
RP 3: shift (um) x = 0.0 +- 0.0, y = 3490.3 +- 16.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 23: shift (um) x = 0.0 +- 0.0, y = 4209.1 +- 15.8, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 103: shift (um) x = 0.0 +- 0.0, y = 2751.2 +- 16.1, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 123: shift (um) x = 0.0 +- 0.0, y = 3417.9 +- 23.9, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0


final merged results:
RP 3: shift (um) x = -3706.2 +- 19.8, y = 3490.3 +- 16.0, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 23: shift (um) x = -41673.8 +- 20.6, y = 4209.1 +- 15.8, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 103: shift (um) x = -2783.2 +- 18.8, y = 2751.2 +- 16.1, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
RP 123: shift (um) x = -41756.8 +- 19.5, y = 3417.9 +- 23.9, z = 0.0 +- 0.0, rotation (mrad) x = 0.0 +- 0.0, y = 0.0 +- 0.0, z = 0.0 +- 0.0
```

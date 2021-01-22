# PPSAlignmentWorker
## Config example:
[`ppsAlignmentWorker_cfi.py`](python/ppsAlignmentWorker_cfi.py)
## Parameters:
| Name        | Type           | Description                                                              |
|-------------|----------------|--------------------------------------------------------------------------|
| `tagTracks` | `cms.InputTag` | Should be set to `"ctppsLocalTrackLiteProducer"`.                      |
| `folder`    | `cms.string`   | Should be the same as the `folder` parameter in DQM configuration.       |
| `label`     | `cms.string`   | Label for EventSetup                                                     |
| `debug`     | `cms.bool`     | When set to `True`, the worker will produce some extra debug histograms. |

# PPSAlignmentHarvester
## Config example:
[`ppsAlignmentHarvester_cfi.py`](python/ppsAlignmentHarvester_cfi.py)
## Parameters:
| Name     | Type         | Description                                                                           |
|----------|--------------|---------------------------------------------------------------------------------------|
| `folder` | `cms.string` | Should be the same as the `folder` parameter in DQM configuration.                    |
| `debug`  | `cms.bool`   | When set to `True`, the harvester will produce an extra ROOT   file with debug plots. |

# Event Setup
Default values come from the `fillDescriptions` method in `CalibPPS/ESProducers/plugins/PPSAlignmentConfigESSource.cc`.<br>
NB: Parameters here are written in snake_case. Many of them are in camelCase in the code (as PPSAlignmentConfig getters).
| Name                   | Type          | Default                                | Description                                                                                                  |
|------------------------|---------------|----------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `debug`                | `cms.bool`    | `False`                                | When set to `True`, the ESProducer will produce an extra ROOT file with   debug plots (from reference run).  |
| `label`                | `cms.string`  | `""`                                   | label to distinguish reference and test fill configs. Should be set   either to `""` (test) or `"reference"` |
| `sequence`             | `cms.vstring` | empty vector                           | Determines order of the alignment methods: `"x_alignemnt"`,   `"x_alignment_relative"`, `"y_alignment"`.     |
| `results_dir`          | `cms.string`  | `"./alignment_results.txt"`            | Directory of a file with the results. If empty (`""`), the file   will not be created.                       |
| `sector_45`            | `cms.PSet`    | [details below](#Sector-config)        | Configuration of sector 45. [Details below](#Sector-config)                                                  |
| `sector_56`            | `cms.PSet`    | [details below](#Sector-config)        | Configuration of sector 56. [Details below](#Sector-config)                                                  |
| `x_ali_sh_step`        | `cms.double`  | `0.01`                                 | Step for x alignment algorithm                                                                               |
| `y_mode_sys_unc`       | `cms.double`  | `0.03`                                 | Squared is an element of y mode uncertainty in y alignment.                                                  |
| `chiSqThreshold`       | `cms.double`  | `50.`                                  | Chi-square threshold of y mode                                                                               |
| `y_mode_unc_max_valid` | `cms.double`  | `5.`                                   | Maximal valid y mode uncertainty                                                                             |
| `y_mode_max_valid`     | `cms.double`  | `20.`                                  | Maximal valid y mode                                                                                         |
| `max_RP_tracks_size`   | `cms.uint32`  | `2.`                                   | Maximal tracksUp or tracksDw size to avoid crowded events                                                    |
| `n_si`                 | `cms.double`  | `4.`                                   | Element of checking whether the cuts passed                                                                  |
| `matching`             | `cms.PSet`    | [details below](#matching)             | Reference dataset parameters. [Details below](#matching)                                                     |
| `x_alignment_meth_o`   | `cms.PSet`    | [details below](#x_alignment_meth_o)   | X alignment parameters. (Details below)[#x_alignment_meth_o]                                                 |
| `x_alignment_relative` | `cms.PSet`    | [details below](#x_alignment_relative) | Relative x alignment parameters. [Details below](#x_aligmment_relative)                                      |
| `y_alignment`          | `cms.PSet`    | [details below](#y_alignment)          | Y alignment parameters. [Details below](#y_alignment)                                                        |
| `binning`              | `cms.PSet`    | [details below](#binning)              | Binning parameters for worker. [Details below](#binning)                                                     |

## Sector config
| Name          | Type         | Default (s_45)                | Default (s_56)                | Description                                          |
|---------------|--------------|-------------------------------|-------------------------------|------------------------------------------------------|
| `rp_N`        | `cms.PSet`   | [details below](#RP-config)   | [details below](#RP-config)   | Near RP configuration. [Details below](#RP-config)   |
| `rp_F`        | `cms.PSet`   | [details below](#RP-config)   | [details below](#RP-config)   | Far RP configuration. [Details below](#RP-config)    |
| `slope`       | `cms.double` | `0.006`                       | `-0.015`                      | Base slope value                                     |
| `cut_h_apply` | `cms.bool`   | `True`                        | `True`                        | If set to `True`, cut_h is applied                   |
| `cut_h_a`     | `cms.double` | `-1.`                         | `-1.`                         | cut_h parameter                                      |
| `cut_h_c`     | `cms.double` | `-38.55`                      | `-39.26`                      | cut_h parameter                                      |
| `cut_h_si`    | `cms.double` | `0.2`                         | `0.2`                         | cut_h parameter                                      |
| `cut_v_apply` | `cms.bool`   | `True`                        | `True`                        | If set to `True`, cut_v is applied                   |
| `cut_v_a`     | `cms.double` | `-1.07`                       | `-1.07`                       | cut_v parameter                                      |
| `cut_v_c`     | `cms.double` | `1.63`                        | `1.49`                        | cut_v parameter                                      |
| `cut_v_si`    | `cms.double` | `0.15`                        | `0.15`                        | cut_v parameter                                      |

### RP config
| Name             | Type         | Default (s_45, rp_N) | Default (s_45, rp_F) | Default (s_56, rp_N) | Default (s_56, rp_F) | Description                                                                                                                     |
|------------------|--------------|----------------------|----------------------|----------------------|----------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `name`           | `cms.string` | `"L_1_F"`            | `"L_2_F"`            | `"R_1_F"`            | `"R_2_F"`            | Name of the RP                                                                                                                  |
| `id`             | `cms.int32`  | `3`                  | `23`                 | `103`                | `123`                | ID of the RP                                                                                                                    |
| `slope`          | `cms.double` | `0.19`               | `0.19`               | `0.40`               | `0.39`               | Base slope value                                                                                                                |
| `sh_x`           | `cms.double` | `-3.6`               | `-42.`               | `-2.8`               | `-41.9`              | Base sh_x value. X alignment method overwrites it.                                                                              |
| `x_min_fit_mode` | `cms.double` | `2.`                 | `2.`                 | `2.`                 | `2.`                 | Mode graph parameter. See [buildModeGraph](plugins/PPSAlignmentHarvester.cc#L648).                                            |
| `x_max_fit_mode` | `cms.double` | `7.`                 | `7.5`                | `7.4`                | `8.`                 | Mode graph parameter. See [buildModeGraph](plugins/PPSAlignmentHarvester.cc#L648).                                            |
| `y_max_fit_mode` | `cms.double` | `7.`                 | `7.5`                | `7.4`                | `8.`                 | Mode graph parameter (in 2018 the same value as x_max_fit_mode). See [buildModeGraph](plugins/PPSAlignmentHarvester.cc#L654). |
| `y_cen_add`      | `cms.double` | `-0.3`               | `-0.3`               | `-0.8`               | `-0.8`               | The value is added to y_cen (mean of y) while constructing a graph in x   alignment.                                            |
| `y_width_mult`   | `cms.double` | `1.1`                | `1.1`                | `1.0`                | `1.`                 | y_width (RMS of y) is multiplied by the value when constructing a graph   in x alignment.                                       |
| `x_slice_min`    | `cms.double` | `7.`                 | `46.`                | `6.`                 | `45.`                | Min x for slice plots (x alignment)                                                                                             |
| `x_slice_max`    | `cms.double` | `19.`                | `58.`                | `17.`                | `57.`                | Max x for slice plots (x alignment)                                                                                             |
| `x_slice_w`      | `cms.double` | `0.2`                | `0.2`                | `0.2`                | `0.2`                | X width for slice plots (x alignment)                                                                                           |

## matching
Should be set in the reference config!
| Name                | Type         | Default           | Description                                                                                                                                                                                     |
|---------------------|--------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `reference_dataset` | `cms.string` | `""`              | Directory of the file with reference dataset histograms. Should be empty   when running the worker for the reference dataset. After that, should be set   to the name of the created ROOT file. |
| `rp_L_F`            | `cms.PSet`   | `-43.` - `-41.`   | Left far RP. Contains two parameters of type `cms.double`: `sh_min` and   `sh_max` - shift range for x alignment                                                                                |
| `rp_L_N`            | `cms.PSet`   | `-4.2` - `-2.4`   | Left near RP. Contains two parameters of type `cms.double`: `sh_min` and   `sh_max` - shift range for x alignment                                                                               |
| `rp_R_N`            | `cms.PSet`   | `-3.6` - `-1.8`   | Right near RP. Contains two parameters of type `cms.double`: `sh_min` and   `sh_max` - shift range for x alignment                                                                              |
| `rp_R_F`            | `cms.PSet`   | `-43.2` - `-41.2` | Right far RP. Contains two parameters of type `cms.double`: `sh_min` and   `sh_max` - shift range for x alignment                                                                               |

## x_alignment_meth_o
| Name                           | Type         | Default        | Description                                                                                                  |
|--------------------------------|--------------|----------------|--------------------------------------------------------------------------------------------------------------|
| `rp_L_F`                       | `cms.PSet`   | `47.` - `56.5` | Left far RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for x alignment   |
| `rp_L_N`                       | `cms.PSet`   | `9.` - `18.5`  | Left near RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for x alignment  |
| `rp_R_N`                       | `cms.PSet`   | `7.` - `15.`   | Right near RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for x alignment |
| `rp_R_F`                       | `cms.PSet`   | `46.` - `54.`  | Right far RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for x alignment  |
| `fit_profile_min_bin_entries`  | `cms.uint32` | `5`            | Minimal number of entries in each bin in fitProfile method                                                   |
| `fit_profile_min_N_reasonable` | `cms.uint32` | `10`           | Minimal number of valid bins in fitProfile method                                                            |
| `meth_o_graph_min_N`           | `cms.uint32` | `5`            | Minimal number of points in each of reference and test graph                                                 |
| `meth_o_unc_fit_range`         | `cms.double` | `0.5`          | Fit range for chi-square graph.                                                                              |

## x_alignment_relative
| Name                   | Type         | Default       | Description                                                                                                           |
|------------------------|--------------|---------------|-----------------------------------------------------------------------------------------------------------------------|
| `rp_L_F`               | `cms.PSet`   | `0.` - `0.`   | Left far RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for relative x alignment   |
| `rp_L_N`               | `cms.PSet`   | `7.5` - `12.` | Left near RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for relative x alignment  |
| `rp_R_N`               | `cms.PSet`   | `6.` - `10.`  | Right near RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for relative x alignment |
| `rp_R_F`               | `cms.PSet`   | `0.` - `0.`   | Right far RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for relative x alignment  |
| `near_far_min_entries` | `cms.uint32` | `100`         | Minimal number of entries in near_far histograms                                                                      |

## y_alignment
| Name                          | Type         | Default        | Description                                                                                                  |
|-------------------------------|--------------|----------------|--------------------------------------------------------------------------------------------------------------|
| `rp_L_F`                      | `cms.PSet`   | `44.5` - `49.` | Left far RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for y alignment   |
| `rp_L_N`                      | `cms.PSet`   | `6.7` - `11.`  | Left near RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for y alignment  |
| `rp_R_N`                      | `cms.PSet`   | `5.9` - `10.`  | Right near RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for y alignment |
| `rp_R_F`                      | `cms.PSet`   | `44.5` - `49.` | Right far RP. Contains two parameters of type `cms.double`: `x_min` and   `x_max` - x range for y alignment  |
| `mode_graph_min_N`            | `cms.uint32` | `5`            | Minimal number of points in mode graph                                                                       |
| `mult_sel_proj_y_min_entries` | `cms.uint32` | `300`          | Minimal number of entries in y projection of multiplicity selection   histograms                             |

## binning
| Name             | Type         | Default       | Description                       |
|------------------|--------------|---------------|-----------------------------------|
| `bin_size_x`     | `cms.double` | `142.3314E-3` | X bin size                        |
| `n_bins_x`       | `cms.uint32` | `210`         | Number of bins in many histograms |
| `pixel_x_offset` | `cms.double` | `40.`         | Pixel x offset                    |
| `n_bins_y`       | `cms.uint32` | `400`         | Number of bins in many histograms |
| `y_min`          | `cms.double` | `-20.`        | Min y for 2D histograms           |
| `y_max`          | `cms.double` | `20.`         | Min y for 2D histograms           |
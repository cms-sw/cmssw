# PPS Global Alignment

## Description

This package is responsible for PPS global alignment, that is alignment of high-luminosity physics runs based on the data from a low-luminosity reference run.

The structure of this software is based on the DQM framework:
1. `PPSAlignmentWorker` - takes tracks of particles produced by `ctppsLocalTrackLiteProducer` as input. It processes them by applying some cuts in order to reduce noise, and uses these data to fill the histograms that will be later used in the next module. Note that filling the histograms can be done in parallel.
2. `PPSAlignmentHarvester` - it analyses the plots filled by the worker, runs the alignment algorithms, and outputs horizontal and vertical corrections.

This package makes use of some modules from other packages:
- `PPSAlignmentConfiguration` in *CondFormats/PPSObjects* - a conditions format used as a configuration input for the alignment procedure,
- `PPSAlignmentConfigurationESSource` in *CalibPPS/ESProducers* - an ESSource module for `PPSAlignmentConfiguration`.

For more information about the alignment algorithms and procedure, detailed instructions on how to run it, and possible use cases check out **[the TWiki page](https://twiki.cern.ch/twiki/bin/viewauth/CMS/PPSAlign)**.

## Contents of the package

- *interface* and *src* - declarations and definitions of auxiliary functions
- *plugins* - implementation of the main modules: `PPSAlignmentWorker` and `PPSAlignmentHarvester`
- *python* - configuration files used by the PPS alignment matrix test and the PCL
- *test* - example configuration files, instructions on how to run them, and expected results

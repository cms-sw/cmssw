### SiStrip fake ESSource to create SiStrip Bad channels from FED errors

A new SiStrip fake source has been added to create Bad Components from the list of Fed detected errors. This
is done using a histogram from DQM output where FedId vs APVId is plotted for detected channels.

The tool inclides following components

 - `SiStripBadModuleFedErrService`: the service defined in `CalibTracker/SiStripESProducers` package which 
   accesses the specific histogram from the DQM root file (to be specified in the configuration) and creates
   `SiStripBadStrip` object checking detected `FedChannel` and `FedId` and using `SiStripFedCabling` information.

    - corresponding configuration is `CalibTracker/SiStripESProducers/python/services/SiStripBadModuleFedErrService_cfi.py`

 - `SiStripBadModuleFedErrRcd` : the record defined in `CalibTracker/Records` package in `SiStripDependentRecords.h`
    and `SiStripDependentRecords.cc`. This is a dependent record and depends on `SiStripFedCablingRcd'. This record
    is filled with `SiStripBadStrip` object.

 - `SiStripBadModuleFedErrFakeESSource` : the actual fake source is defined in the package `CalibTracker/SiStripESProducers` 
   in `plugins/fake` area from the  template 'SiStripTemplateFakeESSource' in `modules.cc`. The record `SiStripBadModuleFedErrRcd`
   is filled with the object `SiStripBadStrip`

    - corresponding configuration file is `CalibTracker/SiStripESProducers/python/fake/SiStripBadModuleFedErrFakeESSource_cfi.py`


 - An overall configuration file can be found in `CalibTracker/SiStripESProducers/test/mergeBadChannel_cfg.py` which merges
   SiStrip Bad channels from PLC, `RunInfo` and SiStripBadModuleFedErrFakeESSource in `SiStripQualityEsProducer` and finally
   listed by the `SiStripQualityStatistics` module.
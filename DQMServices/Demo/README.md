DQMServices/Demo
================


This package provides a few sample modules based on the various plugin types
supported by DQM and has them interact in a dependency-driven way. 

These modules should illustrate how the DQM module types are supposed to work.

In the `test/` folder,  there are instances of all supported module types and
a configuraiton to run them in many different setups. This covers

- Jobs which cover the dataflow at Tier0/ReReco:
  - `DQMEDAnalyzer`s in a (potentially multi-threaded) RECO job with DQMIO output.
  - A merge job with DQMIO input and DQMIO output, like in the production DQMIO MERGE jobs.
  - `DQMEDHarvester`s with DQMIO input and TDriectory output, like in a production HARVESTING job.
  - Similar jobs reading/writing `MEtoEDM` format data.

- Less common offline jobs are not fully covered yet: no multi-run harvesting. 
  Multi-run harvesting is very similar to normal HARVESTING though.

- More exotic configurations:
  - A job with `DQMEDAnalyzer`s writing TDirectory output. This is the classic legacy/commissioning setup.
  - A job with `DQMEDAnalyzer`s using the online TDirectory output.
  - A job with `DQMEDAnalyzer`s using the Protobuf streaming output used in DQM at HLT.
  - A job repacking the Protobuf stream into TDirectory output like `hlt_dqm_clientPB` in online DQM.

- Not covered is live streaming via DQMNet.

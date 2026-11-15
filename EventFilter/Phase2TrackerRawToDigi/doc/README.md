=== Instructions to compile & run outer tracker packer/unpacker  ===

This runs the packer to convert offline clusters to RAW data EDProducts,
and then the unpacker to convert RAW data back to clusters again.
The RAW data is represented by the RawDataBuffer EDProduct,
which is common to all CMS subdetectors.

```
cmsrel CMSSW_20_1_0_pre4
cd CMSSW_20_1_0_pre4/src
cmsenv
git cms-checkout-topic -u P2-Tracker-BES-SW:unpackers_20_1_0_pre4
# Need these as depend on changed TrackerDetToDTCELinkCablingMap.h class
git cms-addpkg CondFormats/DataRecord
git cms-addpkg CondCore/SiPhase2TrackerPlugins
scram b -j
cd EventFilter/Phase2TrackerRawToDigi/test/

-- To process MC

To add to a MC dataset the result of running the cluster --> RAW --> cluster sequence, use:
```
cmsRun PackerAndUnpackerProducer_cfg.py*
```

You can produce a TTTree of the original clusters using the job ClusterAnalyzer_cfg.py, and do the same for the clusters created from
the pack + unpack sequence by rerunning the same job with variable ANALYZE_PACKUNPACK = True.

Then use ROOT macro CompareClusters.C to check if the two sets of clusters
are consistent.

-- Processing real RAW data from Cosmic Rack

Run PackerAndUnpackerProducer_cfg.py followed by ClusterAnalyzer_cfg.py,
with variables UNPACK_CRACK and ANALYZE_CRACK = True. 

-- Rebase Instructions

If you have a personal branch of this code, and wish to update it with changes made by others to tomalin:masterP2TrackerUnpackers , then in new project area:

```
git cms-checkout-topic -u P2-Tracker-BES-SW:unpackers_20_1_0_pre4
git cms-rebase-topic -u myFork:myBranch
```

=== Converting data RAW binary files to a CMSSW dataset

The DAQ system writes a binary file containing the RAW data. You can examine these with the linux command (which prints the 16b in each 128b raw data word in reverse order):

```
hexdump -C
```

The binary file must be converted to a CMSSW dataset containing a RawDataBuffer EDProduct. The code to do this is common to all CMS sub-detectors. It is run with:

```
cmsRun  EventFilter/Utilities/test/unittest_FU_daqsource.py daqSourceMode=DTH buBaseDir=myDTHdataDir/ramdisk/ numFwkStreams=1
```

More details can be found in EventFilter/Utilities/doc/README-DTH.md ,
and also in Ian Tomalin's slides from https://indico.cern.ch/event/1531779/ .

You can examine the contents of the resulting CMSSW dataset with

```
cmsRun DataFormats/FEDRawData/test/DumpRawDataBuffer_cfg.py
```

== Creating simulated RAW binary files

You can create your own binary file containing RAW data, where the data consists of a few test vectors. This can be used to test the unittest_FU_daqsource.py job. You create the binary file with:

```
cmsRun startBU.py fffBaseDir=myDTHdataDir maxLS=1 fedMeanSize=128 eventsPerFile=256 frdFileVersion=0 dataType=DTH
```

=== Instructions to compile & run outer tracker packer/unpacker  ===

This runs the packer to convert offline clusters to RAW data EDProducts,
and then the unpacker to convert RAW data back to clusters again.
The RAW data is represented by the RawDataBuffer EDProduct,
which is common to all CMS subdetectors.

```
cmsrel CMSSW_16_0_0
cd CMSSW_16_0_0/src
cmsenv
git cms-checkout-topic -u P2-Tracker-BES-SW:unpackers_16_0_0
# Need these as depend on changed TrackerDetToDTCELinkCablingMap.h class
git cms-addpkg CondFormats/DataRecord
git cms-addpkg CondCore/SiPhase2TrackerPlugins
scram b -j
cd EventFilter/Phase2TrackerRawToDigi/test/
# Run cluster --> RAW --> cluster sequence for Outer Tracker.
cmsRun PackerAndUnpackerProducer_cfg.py*
```

You can analyzer the clusters before and after this sequence using
the job ClusterAnalyzer_cfg.py, which makes a TTree of them.

-- Rebase Instructions

If you have a personal branch of this code, and wish to update it with changes made by others to tomalin:masterP2TrackerUnpackers , then in new project area:

```
git cms-checkout-topic -u P2-Tracker-BES-SW:unpackers_16_0_0
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

=== Instruction to run the alpaka based unpacker

A GPU version of the unpacker code that uses alpaka exists. To run it, log in to a machine where a GPU is available, then from the CMSSW/src directory:

```
cd EventFilter/Phase2TrackerRawToDigi/test/
cmsRun PackerAndUnpackerProducer_alpaka_cfg.py
```

This runs the packing and unpacking steps, where unpacking is run both with only CPU based unpacker labeled in the file as Unpacker process and the alpaka unpacker labeld as alpakaUnpacker process.
Additionally, this cfg.py file includes a flag `Legacy_Format`, which if you set it to True, will also run the conversion step from the `SoA` output product of the alpaka unpacking EDProducer to the `edmNew::DetSetVector` which is the output product of the only CPU based unpacking EDproducer. Furthermore, it runs in parallel the non-alpaka code (outputting a different EDProduct label) for comparison.

--- Note: 
If the alpaka based code is run on a machine with no GPU, it still runs but only using the serial backend of alpaka which only involves CPUs.   



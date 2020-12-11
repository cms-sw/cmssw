# L1Trigger/CSCTriggerPrimitives Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Code](#code)
3. [Collections](#collections)
4. [LCT	timing](#lct-timing)
5. [History](#history)

## Introduction

(Taken from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCSCTPGEmulator + additions)

The CSC local trigger uses the six-layer redundancy of the CSCs to measure precisely the phi coordinate and to find the correct bunch crossing for a track. Muon segments, also known as Local Charged Tracks (LCT), are found separately in the nearly orthogonal cathode and anode projections by somewhat different algorithms and by different electronics boards; the cathode electronics design is optimized to measure the phi value and the direction with high precision, while the anode electronics design is optimized to determine the muon bunch crossing and the eta value. Up to two cathode and two anode LCTs (referred to as ALCTs and CLCTs) can be found in each chamber during any bunch crossing. The two projections are then combined into three-dimensional LCTs (also referred to as "correlated" LCT) by a timing coincidence in the Trigger Mother Board (TMB). The Muon Port Card (MPC) receives the LCTs from all of the TMB cards in one 60-degree azimuthal sector of one endcap muon station (30-degree subsector in station 1) and relays them over optical fiber links to the Endcap Muon Track Finder (EMTF) and Overlap Muon Track Finder (OMTF). During Run-1 and Run-2 (2015), the MPC would sort all 18 LCTs in a trigger sector and send the best 3 to the former CSC Track-Finder (CSCTF). As of Run-2 (2016) the OMTF and EMTF replaced the old CSCTF. Currently, the MPC does not sort or select LCTs although the functionality remains available in the code.

The CSC Trigger Primitives emulator simulates the functionalities of the anode and cathode LCT processors, of the TMB, and of the MPC.

Starting 2013, studies started to simulate a trigger system where GEM and CSC information is combined in the ME1/1 and ME2/1 station to produce "integrated  LCTs". Integrated LCTs may be built an ALCT, a CLCT, or an ALCT and CLCT, and additional GEM hits. Integrated LCTs are foreseen during CMS Run-3 and beyond (Phase-2).

In the past, integrated LCTs were also constructed from CSC and RPC information in the ME3/1 and ME4/1 station. This was the case in older versions of the code (see for example [here](https://github.com/cms-sw/cmssw/blob/CMSSW_9_4_X/L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME3141RPC.h)). It was however removed in 2017 (see [here](https://github.com/cms-sw/cmssw/commit/968ef1266e0842755918f1a3f58cff7f0e3ae0f8#diff-036d6bb846231665c960705996ca9487) because it is not included in the design of the Phase-2 muon system. The ME3/1 and ME4/1 TMBs were replaced with an upgraded version similar to the ME1/1 upgrade TMB (see [here](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME3141.h)).

## Code

The C++ code is located in `L1Trigger/CSCTriggerPrimitives/src/` and `L1Trigger/CSCTriggerPrimitives/plugins/`. Python configuration can be found in `L1Trigger/CSCTriggerPrimitives/python/`. Utilities to test the code can be found in `L1Trigger/CSCTriggerPrimitives/test/`

The `plugins/` directory contains the producer module `CSCTriggerPrimitivesProducer`. The `CSCTriggerPrimitivesReader` should be used for firmware vs emulator comparisons.

The `src/` directory contains the builder `CSCTriggerPrimitivesBuilder`, processors (`CSCAnodeLCTProcessor`,  `CSCCathodeLCTProcessor`, `GEMCoPadProcessor`), motherboards (`CSCMotherboard` and similar names), a muon port card (`CSCMuonPortCard`), auxiliary classes to produce and access look-up-tables (`CSCUpgradeMotherboardLUT` and `CSCUpgradeMotherboardLUTGenerator`). Trigger patterns are stored in `CSCPatternBank`.

The `CSCTriggerPrimitivesBuilder` instantiates the TMBs for each chamber and the MPCs for each trigger sector. TMBs and MPC are organized in trigger sectors. A trigger sector has 9 associated TMBs and a single MPC. Once instantiated, the TMBs are configured and run according to settings defined `cscTriggerPrimitiveDigis_cfi` (see Configuration). After running the TMB the ALCT/CLCT/LCT collections are read out and put into the event. After all TMBs are run, the MPCs produce LCTs to be sent to the OMTF and EMTF.

The processors `CSCAnodeLCTProcessor` and `CSCCathodeLCTProcessor` produce ALCTs and CLCTs from anode wire-group digis and comparator digis respectively. Typically at least 4 out of 6 layers must have coincident anode/cathode hits to produce an ALCT/CLCT. However, the minimum number of layers is configurable.

The motherboards `CSCMotherboard` ( and similar names) produce LCTs from temporally matched ALCTs and CLCTs. While a TMB can produce several matches (LCT1, LCT2, LCT3,...), only the two highest ranking are sent to the MPC downstream. `CSCUpgradeMotherboard` is the base class for any motherboard commissioned during Run-2 (end of 2018) and beyond. `CSCMotherboardME11` is a derived class for ME1/1 TMBs that run an upgraded algorithm optimized for high-pileup. `CSCGEMMotherboard` is a derived class for TMBs running a GEM-CSC integrated trigger. There are two implementations `CSCGEMMotherboardME11` and `CSCGEMMotherboardME21` for GE1/1-ME1/1 and GE2/1-ME2/1 respectively. The algorithms are based on `CSCMotherboardME11`, but also use GEM information.

The `CSCMuonPortCard` class collects LCTs from a trigger sector and relays them to the OMTF and EMTF. In the past, it would also sort and select the best 3 (out of 18), but that is no longer done. All LCTs are sent to the OMTF and EMTF.

The `CSCUpgradeMotherboardLUTGenerator` and `CSCUpgradeMotherboardLUT` produce and contain look-up-tables that are used in the CSC upgrade algorithm and/or the GEM-CSC algorithm.

A new class is `CSCComparatorCodeLUT` which provides access to look-up-tables for improved local bending and position in Run-3 (CCLUT). Better LCT position and bending is critical to reconstruct L1 muons for displaced signatures, one of the cornerstones of the Phase-2 muon upgrade.

The `test/` directory contains python configuration to test the CSC local trigger and analyzer the data.

## Configuration

The configuration for the CSC local trigger is `cscTriggerPrimitiveDigis_cfi`. By default it is integrated into the standard L1 sequence which can be found [here](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TMuon/python/simDigis_cff.py#L15-L20). When emulating ALCTs/CLCTs/LCTs on simulated wire and comparator digis, the inputs are `"simMuonCSCDigis:MuonCSCWireDigi"` and `"simMuonCSCDigis:MuonCSCComparatorDigi"` respectively. When re-emulating ALCTs/CLCTs/LCTs from unpacked wire and comparator digis - from real LHC data, or GIF++ test-beam data - the inputs are `"muonCSCDigis:MuonCSCWireDigi"` and `"muonCSCDigis:MuonCSCComparatorDigi"` respectively. The configuration of the CSC local trigger as part of the standard L1 sequence on data can be found [here](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/Configuration/python/ValL1Emulator_cff.py#L88-L96).

Besides the input collections, the configuration has parameter sets for the ALCT processors (`alctParam`), CLCT processors (`clctParam`), GEM copad processors (`copadParam`), TMBs (`tmbParam`) and MPCs (`mpcRun2`). A parameter set for common settings is also available (`commonParam`). Parameter sets for upgraded versions of algorithms (more suited for high-luminosity conditions than older versions) are given a label `Phase2` typically. Through the `eras` formalism the upgrades can be switched on.

The Run-2 era (`run2_common`) customizes the default algorithm for updates carried out during Long Shutdown 1. The main difference between Run-1 and Run-2 for the trigger is the unganging of strips in the ME1a subdetector.

The Run-3 era (`run3_GEM`) turns on the GEM-CSC integrated local trigger for the GE1/1-ME1/1 system. For more information, check the [GEM TDR](https://cds.cern.ch/record/2021453/). In addition it turns on the upgrade algorithm for ME1/1 which has improved ALCT/CLCT processor and improved TMB performance under high-luminosity. When running the GEM-CSC integrated local trigger you can use either single GEM pads (`"simMuonGEMPadDigis"`) or clusters or GEM pads (`"simMuonGEMPadDigiClusters"`). The latter emulates how the GEM electronics systems will ship data from the GEM optohybrid board to CSC OTMB. By default the ME1/1 OTMB does not use clusters.

Finally, the Phase-2 era (`phase2_muon`) turns GEM-CSC integrated local trigger for the GE2/1-ME2/1 system and relies on upgraded ALCT/CLCT processors and TMBs for ME3/1 and ME4/1 which will also be impacted by the LHC high-luminosity in Run-4 and beyond. Slight modifications in the settings of processors for ME2/1, ME3/1 and ME4/1 allow to produce higher efficiency stubs.

Note that the configuration can also be taken from the Conditions Database. A dedicated package `L1TriggerConfig/L1CSCTPConfigProducers` contains a producer to do this. However! At the moment, the trigger does **not** take the configuration from the Cond DB. This may change in the future.

## Collections

The producer (`CSCTriggerPrimitivesProducer`) produces several collections:
- `CSCALCTDigiCollection`: ALCTs produced by the [ALCT processor](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/CSCTriggerPrimitives/src/CSCAnodeLCTProcessor.h)
- `CSCCLCTDigiCollection`: CLCTs produced by the [CLCT processor](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/CSCTriggerPrimitives/src/CSCCathodeLCTProcessor.h)
- `CSCCLCTPreTriggerCollection`: a vector for each event contains BX times at which CLCT pre-triggers were recorded.
- `CSCCLCTPreTriggerDigiCollection`: the actual CLCT pre-trigger digis produced by the [CLCT processor](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/CSCTriggerPrimitives/src/CSCCathodeLCTProcessor.h). CLCT pre-trigger digis have a similar dataformat as CLCT trigger digis, except that they require only 3 layers hit as opposed to 4 for triggers.
- `CSCCorrelatedLCTDigiCollection`: LCTs produced by the [TMB](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h) (upgrade TMBs have similar names)
- `CSCCorrelatedLCTDigiCollection (MPCSORTED)`: LCTs produced by the [MPC](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/CSCTriggerPrimitives/src/MuonPortCard.h).
- `GEMCoPadDigiCollection`: coincidences of two GEM pads found in GE1/1 or GE2/1. GEM coincidence pads can be matched spatially and temporally to produce "integrated LCTs".

These collections are used in other CMSSW modules. In particular, the `CSCALCTDigiCollection`, `CSCCLCTDigiCollection`, `CSCCLCTPreTriggerCollection`, `CSCCorrelatedLCTDigiCollection` and `CSCCorrelatedLCTDigiCollection (MPCSORTED)` are used in the "packer" (see [here](https://github.com/cms-sw/cmssw/blob/master/EventFilter/CSCRawToDigi/python/cscPacker_cfi.py#L4-L10)), validation and DQM modules The `CSCCorrelatedLCTDigiCollection (MPCSORTED)` is also used in the OMTF and EMTF emulators to build L1 muon candidates.

## LCT timing

A important property of ALCTs/CLCTs/LCTs is the timing, given by the bunch crossing number (BX). Each of their associated data formats in CMSSW has members to store, set and access the timing. The BX distribution of ALCTs/CLCTs/LCTs is typically not single peak, instead it has shoulders (at earlier BX and later BX) that correspond to ALCTs/CLCTs/LCTs out of time. This shoulders tend to broaden with higher LHC pileup. Note that ALCTs have a better timing than CLCTs. The LCT timing is derived from the ALCT timing in most cases. The central timing (BX0) for each object has a well-defined value in the CSC trigger firmware:
* ALCT: BX0 = 3
* CLCT: BX0 = 7
* LCT: BX0 = 8

The emulator in CMSSW uses the same convention. However, for a long time the emulator would set BX0 to 6 for ALCT, CLCT and LCT. The convention in CMSSW was changed in this [pull request](https://github.com/cms-sw/cmssw/pull/22288) for the entire trigger system. In older CMSSW  releases (e.g. CMSSW_10_1_X and earlier) you will find ALCTs, CLCTs and LCTs at BX0 = 6.

ALCTs constructed with the ALCT processor have BX0=8 throughout the simulation. However, right before they are read out, the BX0 is shifted by -5 BX to BX0=3 so that they agree with the firmware. CLCTs constructed with the CLCT processor have BX0=7. However, when they are used in the `CSCMotherboard` to be correlated to ALCTs, they are shifted by +1 BX to BX0=8. That ensures that ALCT and CLCT timing follow the same definition in the correlation algorithm. LCTs constructed with the `CSCMotherboard` have BX0=8.  The central timing, among various other system settings, is hard-coded in CSCConstants.h.

## History

According to archives, Benn Tannenbaum (UCLA) wrote the CSC local trigger in 1999, based on code by Nick Wisniewski (Caltech) and a framework by Darin Acosta (UFL).

Jason Mumford and Slava Valouev made numerous improvements in 2001. Slava Valouev was the primary developer of the CSC local trigger between 2001 and 2012.

Between 2009 and 2013 Vadim Khotilovich (TAMU) made numerous improvements to make the algorithms more robust in high-pileup environments. These algorithms formed the baseline for the future GEM-CSC algorithms among others.

During LS1 (2015) Slava Krutelyov (then TAMU) improved the code so that the ME1a strips were unganged starting Run-2.

Between 2013 and 2019, Sven Dildick (TAMU) and Tao Huang (TAMU) developed the GEM-CSC integrated trigger, and maintained and improved the CSC local trigger. Among the improvements include bugfixes for Run-2 data emulation (https://github.com/cms-sw/cmssw/pull/14391), shifting the ALCT/CLCT/LCT timing to match the firmware (https://github.com/cms-sw/cmssw/pull/22288), producing CLCT pre-triggers (https://github.com/cms-sw/cmssw/pull/22049) for trigger pattern studies.

Recent work includes:
* Development of the GEM-CSC integrated trigger (lots of PRs)
* The removal of outdated classes `CSCTriggerGeometry` and `CSCTriggerGeomManager` (https://github.com/cms-sw/cmssw/pull/21655)
* The removal of a lot of outdated code (https://github.com/cms-sw/cmssw/pull/24171), which was used for very early MC studies (early 2000s) or to emulate beam-test data from the Magnet Test and Cosmic Challenge (MTCC - 2006). In case you need to look back at this code, please look at the [CMSSW_10_2_X branch](https://github.com/cms-sw/cmssw/tree/CMSSW_10_2_X/L1Trigger/CSCTriggerPrimitives).
* The removal of an outdated DQM module (https://github.com/cms-sw/cmssw/pull/24196).
* The removal of pre-2007 code (https://github.com/cms-sw/cmssw/pull/24254)
* Implementation of a common baseboard class (https://github.com/cms-sw/cmssw/pull/24403)
* Update GE1/1-ME1/1 and GE2/1-ME2/1 local triggers (https://github.com/cms-sw/cmssw/pull/27957, https://github.com/cms-sw/cmssw/pull/28334, https://github.com/cms-sw/cmssw/pull/28605)
* Improvements to the CLCT algorithm following data vs emulator studies Summer and Fall of 2018 (https://github.com/cms-sw/cmssw/pull/25165)
* Implementation of the CCLUT algorithm (https://github.com/cms-sw/cmssw/pull/28044, https://github.com/cms-sw/cmssw/pull/28600, https://github.com/cms-sw/cmssw/pull/29205, https://github.com/cms-sw/cmssw/pull/28846)
# Global Track Trigger
This section describes several componenets of the Global Track Trigger, the CMS subsystem downstream of Level-1 Track Finding, which performs Level-1 vertex-finding for the L1 Correlator Layer 1 (for PF Candidate / PUPPI reconstruction) and globally builds track-only objects such as Jets, HT, MET, mesons (including $\phi$ and $\rho$), etc to be sent to the Global Trigger.

## Data Flow Overview

The current design of the GTT involves several steps. Universally, a GTTInputConversion step which performs such conversions as $\frac{1}{R} to $p_T$ and $tan(\lambda)$ to $\eta$, occurs first. Currently in emulation, this takes information from the 96-bit [TrackWord](https://github.com/cms-sw/cmssw/blob/master/DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h) and overwrites the corresponding fields with the GTT converted values, leaving the Track in a non-canonical state. In [firmware](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/DataFormats/Track?ref_type=heads), the HLS code for the Track [struct](https://gitlab.cern.ch/GTT/LibHLS/-/blob/master/DataFormats/Track/interface/Track.h?ref_type=heads) represents this non-canonical state as separate fields, rather than a single 96-bit word. Afterward InputConversion, multiple TrackSelection (TS) modules are configured/instantiated, potentially 1 for each downstream algorithm, such as VertexFinding (VF), Displaced Vertexing, TrackJets (and thus TrackHT/TrackMissingHT), TrackMET, mesons, $W~\it{to}~3\pi$, and so on. The VertexFinder takes selected tracks and uses (as baseline/extension) a histogramming method to identify the Primary Vertex (PV), weighted either with track $p_T$ (baseline) or a Neural Net score (extended). Downstream, multiple modules of TrackVertexAssociation (TVA) are run (in the baseline, a simple cut-based algo, and in extended algorithm/E2E, a track-association network discriminant cut), taking selected tracks from an appropriate TS modules, the PV from VF, and outputting vertex-associated tracks. These are inputs to JetFinding (JF), meson finding, MET, and other algorithms. In firmware, the outputs from vertex-finding are streamed to the L1 Correlator Layer 1, and all algorithms (including vertex-finding) outputs are sent to the L1 Global Trigger.

### GTT Input Conversion
Input conversion handles the change from 1/R (really q/R) to $p_T$ and $tan(\lambda)$ to $\eta$. In LibHLS, this is controlled with a few constants that denote how many integer and fixed-float bits should be used. In emulation, constants are located inside the GTT plugin under ConversionBitWidths.
LibHLS $\eta$ [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/ConversionEta?ref_type=heads) and $p_T$ [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/ConversionPt?ref_type=heads)
CMSSW [plugin](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/plugins/L1GTTInputProducer.cc)

#### Status (October 2024)
The Input conversion in Firmware centrally stores conversion constants and consistently uses the central GTT format (struct) for tracks. Meanwhile, the emulator is in a somewhat different spot, with a low-granularity conversion of $p_T$ with 7 integer bits and 3 float bits (currently consistent with firmware). For reading the $p_T$ back, different algorithms make different assumptions about the width of the datatype, either the actual ap_fixed<10,7> with 7 integer bits, or ap_fixed<14,9>. This works because of a bit-shift that 0-pads the 2 least significant bits of the actual ap_fixed<10,7> data stored in the previoius canonical L1T $\frac{1}{R}$ word; ergo the least significant integer bits are aligned under either interpretation. The $p_T$ conversion also uses an 8-bit LUT to map track values from the ap_int datatype to ap_fixed, which means that the least significant bits are ignored, dropping 2**2 in resolution. Because of the non-linear conversion, this produces strong discretization of high $p_T$ tracks. A possible remediation of this would involve either splitting the LUT so that a more granular version is applied to high $p_T$ tracks, or equivalently residual/correction-LUTs to adjust the response where needed. Either approach should require a modest increase in LUT resources but improve the conversion loss. Additionally, studies on $\phi$ mesons and reconstructing $B_s$ from them indicates that the $\eta$ conversion is a significant source of error, but also the $p_T$ appears to have biases (which may have a charge dependence). The latter indicates that there may be an inconsistent use of half-bin shifting in calculations somewhere. The $\eta$ conversion uses a 128-sized LUT (owing to symmetry in the transform, it's cut in half relative to a simple calculation), into an ap_fixed<8,3> datatype; the LUT should be increased by a factor of 2^3 at least to ap_fixed<11,3>, improving the granularity significantly for $\phi$ meson reconstruction.

### Vertex Finder (VF)
#### FastHisto (FH) - Baseline Algorithm
This version of the algorithm serves as the baseline from TDR studies. This uses a histogram (256 bins as of 2023) which is filled with track $p_T$ as weights. A single vertex is chosen by a sliding-window algorithm which finds the consecutive bins (currently 3) containing the maximum $p_T$-sum (using a flat kernel). An inversion LUT which is built to store the mapping from $p_T$ to $\frac{1}{p_T}$ is used in combination with a window-bin-indexed weighted-sum of per-bin $p_T$-sums is used to calculate the $p_T$-weighted location of the peak, and this is stored as the vertex $z_0$ position. The $p_T$-sum from the window is denoted as the sumPt of the vertex.
LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/Vertex?ref_type=heads)
VertexFinder class [interface](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/VertexFinder/interface/VertexFinder.h), [src](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/VertexFinder/src/VertexFinder.cc), and [plugin](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/VertexFinder/plugins/VertexProducer.cc) with CMSSW default configuration [here](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/VertexFinder/python/l1tVertexProducer_cfi.py). The Simulation tag is 'l1tVertexFinder' whereas the emulator is usually grabbed via 'l1tVertexFinderEmulator.'
#### End-to-end Neural Network (E2E) - Extended Algorithm
The End-to-end Neural Network is a beyond-baseline version of the algorithm. It uses the same structure as FastHisto, but replaces weighting tracks by their $p_T$ with a DNN discriminant score. This produces an improved vertex-finding efficiency and resolution over FH. This is trained in tandem with TrackAssociationNetworks to replace the cut-based TrackSelection algorithm typically paired with FH.

#### Status as of October 2024
The FastHisto (Emulation) algorithm has bit-level agreement with the firmware using several thousand events from $t\bar{t}$ simulation (200 PileUp). The algorithm can handle mulitple vertices. In LibHLS/firmware, only one vertex is enabled to reduce resource usage. An implementation detail is that the $p_T$ sum per bin is calculated untruncated in both firmware and emulation, but prior to the vertex-finding portion (sliding window algo), the precision is reduced. Of the fields proposed for the vertex, only the valid bit, the sumPt, and the $z_0$ position are filled; all other fields, including the quality field, the nTracks in/out PV are 0-filled. A design shortcoming is that the number of bins is a compile-time constant, but runtime-configurable constants passed into the configuration as `FH_HistogramParameters` indicate the minimum and maximum $z$ position for the histogram, and the 3rd parameter must divide this range appropriately to match the number of bins in firmware. For example, `FH_HistogramParameters = cms.vdouble(-20.46912512, 20.46912512, 0.15991504)` is appropriate for 256 bins as currently used.

### Track Selection (TS)

In CMSSW GTT emulation the [plugin](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/plugins/L1TrackSelectionProducer.cc) is defined within the L1TTrackMatch subpackage, and configured with default settings [here](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/python/l1tTrackSelectionProducer_cfi.py). In firmware, the HLS code is concentrated in this LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/TrackSelection?ref_type=heads)

#### Status as of October 2024
The TrackSelection is partially desynchronized with the Firmware at HLS level. The firmware currently runs 3 duplicates of TS for VF, JF, and MET. The emulation default is not necessarily synchronized (per algorithm), and some of the capabilities in Emulation are not yet propagated back to HLS firmware code, notably for any track MVA quality cuts.

### Track Vertex Association (TVA) - Baseline (cut-based) Implementation
CMSSW [plugin](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/plugins/L1TrackVertexAssociationProducer.cc)
CMSSW [config](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/python/l1tTrackVertexAssociationProducer_cfi.py)
LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/TrackVertexAssociation?ref_type=heads)

#### Status as of October 2024
The Baseline TVA module is extremely simple, primarily cutting on the $\Delta~z$ between the PV and the selected tracks from the upstream TS module.

### Jet Finding (JF)
Jet finding uses a 2-layer clustering algorithm. The firmware is written in Verilog, and the emulator can be found in CMSSW in the central [plugin](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/plugins/L1TrackJetEmulatorProducer.cc), [configuration](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/python/l1tTrackJetsEmulation_cfi.py), and common [header](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/plugins/L1TrackJetClustering.h)

#### Status as of October 2024
The output of the emulator does not exactly match the firmware currently. An ongoing PR exists to fix several discrepancies, including a missing valid bit in emulation, and the binning of the $\eta-\phi$ plane.

### HT and Missing HT (HT)
The HT / Missing HT modules only exist in Emulation currently, with plans for an HLS implementation in Firmware in the near future. The CMSSW [plugin](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/plugins/L1TkHTMissEmulatorProducer.cc) and associated [configuration](https://github.com/cms-sw/cmssw/tree/master/L1Trigger/L1TTrackMatch/python)

#### Status as of October 2024
The (Missing)HT Emulator currently uses a centrally defined [data format](https://github.com/cms-sw/cmssw/blob/master/DataFormats/L1Trigger/interface/EtSum.h) which only permits storing either the scalar or vector-sum in a (potentially) bit-accurate way. In other subsystems (e.g. L1Calo), this is addressed by storing two variations of the datatype, one where the scalar sum is stored in a hardware-accurate way, and another with the vector sum. Currently GTT only stores one copy, and the vector sum is stored as a float, while the scalar sum, phi, and number of jets are stored as integers.

### MET
CMSSW [plugin](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/plugins/L1TrackerEtMissEmulatorProducer.cc)
CMSSW [configuration](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/python/l1tTrackerEmuEtMiss_cfi.py)
LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/MET?ref_type=heads)

#### Status as of October 2024
The MET module has bit-accurate agreement between firmware and emulation. The LUT converting the track global $\phi$ value to a $cos(\phi)$ or $sin(\phi)$ value was reduced to a size of 1024 in order to meet firmware timing constraints. This is accomplished with a single LUT in combination with trigonometric identities to avoid separate $cos(\phi)$ and $sin(\phi)$ LUTs. Similar to the (Missing)HT module, only part of the output is stored in a potentially bit-accurate way in emulation (in this case, the vector sum, being the true MET, along with the phi and number of tracks)

### Phi and Rho Meson Finding
These currently exist as simulation+emulation studies in private CMSSW branches, with ongoing work to create HLS-based firmware modules.

### $W~\it{to}~3\pi$ (W3pi)
CMSSW [TkTriplet](https://github.com/cms-sw/cmssw/blob/master/DataFormats/L1TCorrelator/interface/TkTriplet.h)
CMSSW [plugin](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/plugins/L1TrackTripletEmulatorProducer.cc)
CMSSW [configuration](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/python/l1tTrackTripletEmulation_cfi.py)
LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/WtoThreePi?ref_type=heads)

### GTTFileReader, GTTFileWriter
[reader](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/DemonstratorTools/plugins/GTTFileReader.cc)
[writer](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/DemonstratorTools/plugins/GTTFileWriter.cc)
These plugins make use of the subsystem-agnostic BoardDataReader and BoardDataWriter classes, respectively, specified in the reader's [interface](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/DemonstratorTools/interface/BoardDataReader.h) and [src](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/DemonstratorTools/src/BoardDataReader.cc) files, and likewise for the writer [interface](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/DemonstratorTools/interface/BoardDataWriter.h) and [src](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/DemonstratorTools/src/BoardDataWriter.cc)
The GTT configuration constants are mostly gathered in the [GTTInterface.h](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/DemonstratorTools/interface/GTTInterface.h) which specifies how link inputs/frames map to given collections of objects for pattern-buffer writing and reading. This specified, for example, whether vertices are placed in link 3 or link 0 to GT for a given board, and at which starting frame they may appear. Similarly other collections like jets, displaced jets, MET, HT link and frame position encoding are specified in here.

## Emulation Pattern (Buffer) Files

A git [submodule](https://gitlab.cern.ch/GTT/Data) stores fixed copies of the emulation buffer / pattern files for use in [LibHLS](https://gitlab.cern.ch/GTT/LibHLS/), using git [LFS](https://git-lfs.com/)

### Generating pattern files
A script in the CMSSW L1Trigger/DemonstratorTools/test/gtt folder contains a configuration to export and import pattern file buffers into CMSSW.
[script](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/DemonstratorTools/test/gtt/createFirmwareInputFiles_cfg.py).
Typical usage looks like:
```
cmsRun createFirmwareInputFiles_cfg.py maxEvents=1008 format=APx inputFiles=L1Trigger/DemonstratorTools/python/TT_TuneCP5_14TeV-powheg-pythia8_Phase2Spring23DIGIRECOMiniAOD-PU200_L1TFix_Trk1GeV_131X_mcRun4_realistic_v9-v1_GEN-SIM-DIGI-RAW-MINIAOD_APxBuffers_cff.py
```
An example configuration as referred to via the `inputFiles` kwarg can be found below, in this README.

### Loading tracks and vertices from pattern files
Two input options, `tracks` and `vertices` can be passed 3 options, `donotload`, `load`, and `overwrite` to modify the behavior of patter-file writing. For each collection, the option `donotload` implies that the appropriate collection will be generated or read in from CMSSW collections. The `load` option instead dictates that pattern files' contents, specified in the inputFiles configuration, will be decoded (verifying they can be read back via BoardDataReader class, as configured via GTTFileReader), but the final collections will still be read/generated from CMSSW collections generated upstream in TrackFinding. The final option, `overwrite,` decodes the objects from pattern files and makes these the corresponding collection used in subsequent steps. That is, if `tracks=overwrite` is used, then the tracks passed into the GTTInputConversion and downstream TS, VF, etc. will come from pattern files written in the appropriate `readerformat` (APx, EMPv2, ...). As long as vertices is not set to `overwrite` as well, this will calculate new vertices from these pattern-buffer loaded tracks instead of those from the source root files or generated by the TrackFinder upstream. Similarly, if `vertices=overwrite,` the vertices collection will be read from the pattern files and upstream tracks (regardless of source and `tracks=X` option selected) will be ignored.
```
cmsRun createFirmwareInputFiles_cfg.py maxEvents=1008 format=APx inputFiles=L1Trigger/DemonstratorTools/python/TT_TuneCP5_14TeV-powheg-pythia8_Phase2Spring23DIGIRECOMiniAOD-PU200_L1TFix_Trk1GeV_131X_mcRun4_realistic_v9-v1_GEN-SIM-DIGI-RAW-MINIAOD_APxBuffers_cff.py tracks=overwrite vertices=donotload readerformat=APx
```

An example of the `inputFiles` configuration including pointers to buffer files follows:

```
import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
    '/store/mc/Phase2Spring23DIGIRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU200_L1TFix_Trk1GeV_131X_mcRun4_realistic_v9-v1/50000/005bc30b-cf79-4b3b-9ec1-a80e13072afd.root',
    '/store/mc/Phase2Spring23DIGIRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU200_L1TFix_Trk1GeV_131X_mcRun4_realistic_v9-v1/50000/009bd7ba-4295-46ef-a5bc-9eb3d2cd3cf7.root',
    '/store/mc/Phase2Spring23DIGIRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU200_L1TFix_Trk1GeV_131X_mcRun4_realistic_v9-v1/50000/017a99d2-4636-4584-97d0-d5499c3b453c.root',
    '/store/mc/Phase2Spring23DIGIRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU200_L1TFix_Trk1GeV_131X_mcRun4_realistic_v9-v1/50000/02020287-a16f-41db-8021-f9bcd272f6c9.root',
    '/store/mc/Phase2Spring23DIGIRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU200_L1TFix_Trk1GeV_131X_mcRun4_realistic_v9-v1/50000/02ca41cb-9638-4703-88b7-799c30fd2656.root',
    '/store/mc/Phase2Spring23DIGIRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU200_L1TFix_Trk1GeV_131X_mcRun4_realistic_v9-v1/50000/03171e00-8880-4c09-807a-0c1d5bac2797.root',
    '/store/mc/Phase2Spring23DIGIRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU200_L1TFix_Trk1GeV_131X_mcRun4_realistic_v9-v1/50000/04c836c3-66f4-44d5-a8fd-2faf5e4aa623.root',
] )
correlator_source = cms.Source ("PoolSource",
                                fileNames = cms.untracked.vstring(
    'APx/L1GTTOutputToCorrelatorFile_0.txt',
    'APx/L1GTTOutputToCorrelatorFile_1.txt',
    'APx/L1GTTOutputToCorrelatorFile_2.txt',
    'APx/L1GTTOutputToCorrelatorFile_3.txt',
    'APx/L1GTTOutputToCorrelatorFile_4.txt',
    ...,
    'APx/L1GTTOutputToCorrelatorFile_54.txt',
    'APx/L1GTTOutputToCorrelatorFile_55.txt',
    )
)
track_source = cms.Source ("PoolSource",
                                fileNames = cms.untracked.vstring(
    'APx/L1GTTInputFile_0.txt',
    'APx/L1GTTInputFile_1.txt',
    'APx/L1GTTInputFile_2.txt',
    'APx/L1GTTInputFile_3.txt',
    ...,
    'APx/L1GTTInputFile_60.txt',
    'APx/L1GTTInputFile_61.txt',
    'APx/L1GTTInputFile_62.txt',
    )
)
```
For loading tracks or vertices from buffers, the outputs from the `createFirmwareInputFiles_cfg.py` should be placed in a subfolder labeled `APx` relative to that same script.

### Sidebands in different formats
The APx and EMPv2 have different sideband codes. In LibHLS a script exists which can strip the APx sideband codes from pattern files and write these out as new txt files can be found [here](https://gitlab.cern.ch/GTT/Data/-/blob/master/Scripts/APx/turn_off_sideband.py)
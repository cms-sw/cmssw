# Global Track Trigger
This section describes several componenets of the Global Track Trigger, the CMS subsystem downstream of Level-1 Track Finding, which performs Level-1 vertex-finding for the L1 Correlator Layer 1 (for PF Candidate / PUPPI reconstruction) and globally builds track-only objects such as Jets, HT, MET, mesons (including phi and rho), etc.

## Data Flow Overview

The current design of the GTT involves several steps. Universally, a GTTInputConversion step which performs such conversions as $\frac{1}{R} to $p_T$ and $tan(\lambda)$ to $\eta$. Currently in emulation, this takes information from the 96-bit [TrackWord](https://github.com/cms-sw/cmssw/blob/master/DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h) and overwrites the corresponding fields with the GTT converted values, leaving the Track in a non-canonical state. In [firmware](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/DataFormats/Track?ref_type=heads), the HLS code for the Track [struct](https://gitlab.cern.ch/GTT/LibHLS/-/blob/master/DataFormats/Track/interface/Track.h?ref_type=heads) represents this non-canonical state as separate fields, rather than a single 96-bit word. Afterwards, multiple TrackSelection (TS) modules are configured/instantiated, potentially 1 for each downstream algorithm, such as VertexFinding (VF), TrackJets (and thus TrackHT/TrackMissingHT), TrackMET, mesons, $W\rarrow~3\pi$, and so on. The VertexFinder takes selected tracks and uses (as baseline/extension) a histogramming method to identify the Primary Vertex (PV), weight either with track $p_T$ or a Neural Net score. Downstream, multiple modules of TrackVertexAssociation (TVA) are run, taking selected tracks from one of the TS modules, the PV from VF, and they output vertex-associated tracks. These are inputs to JetFinding (JF), meson finding, MET, and others.

### GTT Input Conversion
FIXME
LibHLS [$\eta$ module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/ConversionEta?ref_type=heads) and [$p_T$ module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/ConversionPt?ref_type=heads)

### Vertex Finder (VF)
#### FastHisto (FH) - Baseline Algorithm
FIXME
LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/Vertex?ref_type=heads)
#### End-to-end Neural Network (E2E) - Extended Algorithm
FIXME

#### Status
The FastHisto algorithm has bit-level agreement with the firmware using several thousand events from $t\bar{t}$ simulation (200 PileUp).

### Track Selection (TS)

In CMSSW GTT emulation the plugin is defined [here](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/plugins/L1TrackSelectionProducer.cc) and configured with default settings [here](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTrackMatch/python/l1tTrackSelectionProducer_cfi.py). In firmware, the HLS code is concentrated in this LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/TrackSelection?ref_type=heads)

#### Status
The TrackSelection is mostly synchronized with the Firmware at HLS level. The firmware currently runs 3 duplicates of TS for VF, JF, and MET. The emulation default is not necessarily synchronized.

### Track Vertex Association (TVA)
FIXME
LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/TrackVertexAssociation?ref_type=heads)

### Jet Finding (JF)
FIXME

### HT and Missing HT (HT)
FIXME

### MET
FIXME
LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/MET?ref_type=heads)

### Phi and Rho Meson Finding
FIXME

### $W\rarrow 3\pi$ (W3pi)
FIXME
LibHLS [module](https://gitlab.cern.ch/GTT/LibHLS/-/tree/master/Modules/WtoThreePi?ref_type=heads)

## Emulation Pattern (Buffer) Files

A git [submodule](https://gitlab.cern.ch/GTT/Data) stores fixed copies of the emulation buffer / pattern files for use in [LibHLS](https://gitlab.cern.ch/GTT/LibHLS/), using git [LFS](https://git-lfs.com/)

### Generating pattern files
FIXME
### Loading tracks and vertices from pattern files
FIXME
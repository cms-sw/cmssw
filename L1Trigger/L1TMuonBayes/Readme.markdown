The package **L1TMuonBayes** contains two producers:  
`plugins/L1TMuonBayesMuCorrelatorTrackProducer.h`  
`plugins/L1TMuonBayesOmtfTrackProducer.h`

#### L1TMuonBayesMuCorrelatorTrackProducer
**L1TMuonBayesMuCorrelatorTrackProducer** is the emulator of the **ttTracks plus muon stubs correlator** in the whole eta raneg (|eta|<2.4). Additionally, it allows for **triggering on HSCPs**. The corresponding classes are in `L1TMuonBayes/interface/MuCorrelator` .

For algorithm description look [here](https://indico.cern.ch/event/791517/contributions/3362988/attachments/1818183/2973006/190326_muon_correlator_CIEMAT.pdf) and [here](https://indico.cern.ch/event/818788/contributions/3420714/subcontributions/280120/attachments/1841157/3018726/190508_muon_correlator_algorithm_meeting.pdf) .

The algorithm matches the tracking trigger tracks (ttTracks) with the muon stubs from DT, CSC, RPC and iRPC. Using of the GEM stubs is no yet implemented. 

L1TMuonBayesMuCorrelatorTrackProducer produces three products with the following instanceNames (defined in `plugins/L1TMuonBayesMuCorrelatorTrackProducer.h lines 40-42`):  
* **AllTracks** - all tracks produced by the muon correlator emulator, without additional cuts,  
* **MuonTracks** - "fast" tracks, i.e. with at least two muon stubs in the same bx as ttTrack (=> not HSCPs), and with some cuts reducing rate. Should be used for "normal" muon trigger algorithms,  
* **HscpTracks** - "slow" tracks, i.e. exclusive versus the "fast" tracks and passing some cuts. Shuld be used for the HSCP trigger algorithms.  

The cuts are defined in the `L1TMuonBayesMuCorrelatorTrackProducer::produce() lines 219-236`  
**Only the MuonTracks and HscpTracks should be used for the menu studies**  

Please note that the HSCPs with beta > 0.7 are mostly reconstructed as MuonTracks, since they cannot be distinguished from the "normal" muons (beta = 1).

The products are of type `l1t::BayesMuCorrTrackCollection` (i.e. `BXVector<BayesMuCorrelatorTrack>`). `BayesMuCorrelatorTrack` is defined in `DataFormats/L1TMuon/interface/BayesMuCorrelatorTrack.h`.

The default producer configuration is in `L1Trigger/L1TMuonBayes/python/simBayesMuCorrelatorTrackProducer_cfi.py`

The emulator needs two files with the data for the LUTs:  
`pdfModuleFile = cms.FileInPath("L1Trigger/L1TMuon/data/muonBayesCorrelator_config/muCorrelatorPdfModule.xml"),`  
`timingModuleFile  = cms.FileInPath("L1Trigger/L1TMuon/data/muonBayesCorrelator_config/muCorrelatorTimingModule.xml"),`

The HSCP triggering can be turned on or off with the paramater 'useStubsFromAdditionalBxs'. E.g. if it is set to 3, then the muon stubs (DT, CSC, RPC, iRPC) from the BX = 0, 1, 2, 3 are matched to the ttTrack, which allow to find the HSCP with beta down to ~0.2-0.3. If it is 0, then only the muon stubs from the BX=0 are used. 

#### L1TMuonBayesOmtfTrackProducer
**L1TMuonBayesOmtfTrackProducer** is the emulator of the OMTF (Overlap Muon Track Finder) trigger - at the moment it works exactly the same as the current OMTF emulator (L1Trigger/L1TMuonOverlap), but it is used to develop the new features for the phase 2. The corresponding classes are in L1TMuonBayes/interface/Omtf.  
**The OMTF emulator in L1TMuonBayesOmtfTrackProducer is in development phase, and should be used only by the OMTF experts**.



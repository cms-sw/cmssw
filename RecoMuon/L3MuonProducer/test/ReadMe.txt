ReadMe for New L3 Configuration
-------------------------------

Information:
------------

The Muon HLT L3 algorithms have been developed between June 2014 and June 2015. The new algorithms contain one Outside-In and one Inside-Out based algorithm. The advances in iterative tracking online has been utilised. Higher performance with respect to the current L3 algorithms has been seen in tests performed. However, we wish the interested community to also test the new algorithms and give feedback to Muon HLT.

More information can be found in the following presentation:
https://indico.cern.ch/event/402284/contribution/2/attachments/805981/1104536/MuonHLT20150616-NewL3Performance.pdf


Instructions:
-------------

The newL3.py is a customiser - which modifies the HLT menu. This means someone can get a menu and add the following two lines to the end of it to switch on the New L3 algorithms:
from newL3 import addL3ToHLT
process = addL3ToHLT(process)

This has been tested when making a menu with the following paths:
HLT_Mu8_v*
HLT_Mu8_TrkIsoVVL_v*
HLT_Mu17_v*
HLT_Mu20_v*
HLT_IsoMu20_v*
HLT_Mu45_eta2p1_v*
HLT_Mu50_v*
HLT_Mu17_Mu8_DZ_v*
HLT_Mu17_TkMu8_DZ_v*
HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*
HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*
HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*
HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*
HLT_DoubleMu23NoFiltersNoVtxDisplaced_v*



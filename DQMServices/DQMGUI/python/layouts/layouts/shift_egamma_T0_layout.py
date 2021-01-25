from .adapt_to_new_backend import *
dqmitems={}

def shiftegammalayout(i, p, *rows): i["00 Shift/Egamma/" + p] = rows

shiftegammalayout(dqmitems, "1-Good Photon Candidates: Et Spectra",
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_04_phoEtBarrel",
	'description': "Transverse energy of good candidate photons in ECAL barrel - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }],
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_04_phoEtEndcaps",
	'description': "Transverse energy of good candidate photons in ECAL endcaps - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "2-Good Photon Candidates: R9",
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_10_r9Barrel",
	'description': "R9 parameter for good candidate photons in ECAL barrel - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }],
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_10_r9Endcaps",
	'description': "R9 paramater for good candidate photons in ECAL endcaps - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>"}])

shiftegammalayout(dqmitems, "3-Good Photon Candidates: SigmaIetaIeta",
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_13_phoSigmaIetaIetaBarrel",
	'description': "SigmaIetaIeta parameter for good candidate photons in ECAL barrel - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }],
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_13_phoSigmaIetaIetaEndcaps",
	'description': "SigmaIetaIeta paramater for good candidate photons in ECAL endcaps - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>"}])

shiftegammalayout(dqmitems, "4-PiZeros",
  [{'path': "Egamma/PiZeroAnalyzer/Pi0InvmassEB",
	'description': "Reconstructed mass of the PiZero particle - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "5-Good Photon Candidates: ECAL Isolation Sum",
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_37_ecalSum",
	'description': "Ecal sum in iso cone - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "6-Good Photon Candidates: HCAL Isolation Sum",
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_42_hcalSum",
	'description': "Hcal sum in iso cone - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "7-Good Photon Candidates: Eta distribution",
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_06_phoEta",
	'description': "Eta Distribution - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }])

shiftegammalayout(dqmitems, "8-Good Photon Candidates: H Over E",
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_47_hOverEBarrel",
	'description': "H over E for good candidate photons in ECAL barrel - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>" }],
  [{'path': "Egamma/stdPhotonAnalyzer/GoodCandidatePhotons/Et above 20 GeV/h_47_hOverEEndcaps",
	'description': "H over E for good candidate photons in ECAL endcaps - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>"}])

shiftegammalayout(dqmitems, "9-Good Photon Candidates: invMass",
  [{'path': "Egamma/stdPhotonAnalyzer/InvMass/h_01_invMassAllIsolatedPhotons",
	'description': "Invariant mass of all photons - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEgamma>Shift Instructions</a>"}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)

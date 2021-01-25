from .adapt_to_new_backend import *
dqmitems={}

def pftaulayout(i, p, *rows): i["RecoTauV/Layouts/" + p] = rows

pftaulayout(
	dqmitems,
	"SingleMu/00ba - Fake rate from muons vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_Matched/PFJetMatchingEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/00bb - Fake rate from muons vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/00aa - Fake rate from muons vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_Matched/PFJetMatchingEffpt', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpt', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/00ab - Fake rate from muons vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/00da - Fake rate from muons vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_Matched/PFJetMatchingEffphi', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffphi', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/00db - Fake rate from muons vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/00ca - Fake rate from muons vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_Matched/PFJetMatchingEffeta', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffeta', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/00cb - Fake rate from muons vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/01b - Muon rejection fake rate vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'MediumMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'TightMuonRejection fake rate'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/01a - Muon rejection fake rate vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEffpt', 'draw': {'drawopts': 'e'}, 'description': 'MediumMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEffpt', 'draw': {'drawopts': 'e'}, 'description': 'TightMuonRejection fake rate'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/01d - Muon rejection fake rate vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEffphi', 'draw': {'drawopts': 'e'}, 'description': 'MediumMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEffphi', 'draw': {'drawopts': 'e'}, 'description': 'TightMuonRejection fake rate'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/01c - Muon rejection fake rate vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEffeta', 'draw': {'drawopts': 'e'}, 'description': 'MediumMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEffeta', 'draw': {'drawopts': 'e'}, 'description': 'TightMuonRejection fake rate'}]
	)
pftaulayout(
	dqmitems,
	"Jet/02ba - Fake rate from jets vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_Matched/PFJetMatchingEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/02bb - Fake rate from jets vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/02aa - Fake rate from jets vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_Matched/PFJetMatchingEffpt', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpt', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/02ab - Fake rate from jets vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/02da - Fake rate from jets vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_Matched/PFJetMatchingEffphi', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffphi', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/02db - Fake rate from jets vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/02ca - Fake rate from jets vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_Matched/PFJetMatchingEffeta', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffeta', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/02cb - Fake rate from jets vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/03a - Distributions of size and sumPt for signalPFCands, muons from Z faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of signalPFCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of signalPFCands'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/03c - Distributions of size and sumPt for isolationPFGammaCands, muons from Z faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of isolationPFGammaCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of isolationPFGammaCands'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/03b - Distributions of size and sumPt for isolationPFChargedHadrCands, muons from Z faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of isolationPFChargedHadrCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of isolationPFChargedHadrCands'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/03d - Distributions of size and sumPt for isolationPFNeutrHadrCands, muons from Z faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus Size of isolationPFNeutrHadrCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'muons from Z faking taus SumPt of isolationPFNeutrHadrCands'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/04a - Distributions of size and sumPt for signalPFCands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of signalPFCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of signalPFCands'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/04c - Distributions of size and sumPt for isolationPFGammaCands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFGammaCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFGammaCands'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/04b - Distributions of size and sumPt for isolationPFChargedHadrCands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFChargedHadrCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFChargedHadrCands'}]
	)
pftaulayout(
	dqmitems,
	"SingleMu/04d - Distributions of size and sumPt for isolationPFNeutrHadrCands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFNeutrHadrCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFNeutrHadrCands'}]
	)
pftaulayout(
	dqmitems,
	"Jet/00ba - Fake rate from jets vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_Matched/PFJetMatchingEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/00bb - Fake rate from jets vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/00aa - Fake rate from jets vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_Matched/PFJetMatchingEffpt', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpt', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/00ab - Fake rate from jets vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/00da - Fake rate from jets vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_Matched/PFJetMatchingEffphi', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffphi', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/00db - Fake rate from jets vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/00ca - Fake rate from jets vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_Matched/PFJetMatchingEffeta', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffeta', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/00cb - Fake rate from jets vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
pftaulayout(
	dqmitems,
	"Jet/01a - Distributions of size and sumPt for signalPFCands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of signalPFCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of signalPFCands'}]
	)
pftaulayout(
	dqmitems,
	"Jet/01c - Distributions of size and sumPt for isolationPFGammaCands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFGammaCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFGammaCands'}]
	)
pftaulayout(
	dqmitems,
	"Jet/01b - Distributions of size and sumPt for isolationPFChargedHadrCands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFChargedHadrCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFChargedHadrCands'}]
	)
pftaulayout(
	dqmitems,
	"Jet/01d - Distributions of size and sumPt for isolationPFNeutrHadrCands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFNeutrHadrCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFNeutrHadrCands'}]
	)
pftaulayout(
	dqmitems,
	"Jet/01e - Distributions of size and sumPt for isolation PF Cands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits_Size_isolationPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Size of isolationPFCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits_SumPt_isolationPFCands', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus SumPt of isolationPFCands'}]
	)
pftaulayout(
	dqmitems,
	"Jet/01f - Distributions of Raw Quantities of Tau Cands, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_vs_ptTauVisible', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus ptTauVisible'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_vs_etaTauVisible', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus etaTauVisible'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_vs_phiTauVisible', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus phiTauVisible'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_TauCandMass', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus Candidate Mass'}]
	)
pftaulayout(
	dqmitems,
	"Jet/01g - Distributions of Tau Cands Multiplicity, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_nTaus_allHadronic', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus multiplicity nTaus_allHadronic'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_nTaus_oneProng0Pi0', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus multiplicity nTaus_oneProng0Pi0'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_nTaus_twoProng0Pi0', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus multiplicity nTaus_twoProng0Pi0'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_nTaus_threeProng0Pi0', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus multiplicity nTaus_threeProng0Pi0'}]
	)
pftaulayout(
	dqmitems,
	"Jet/01h - Distributions of Tau Cands pTRatio, QCD Jets faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_pTRatio_allHadronic', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus pTRatio_allHadronic'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_pTRatio_oneProng0Pi0', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus pTRatio_oneProng0Pi0'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_pTRatio_twoProng0Pi0', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus pTRatio_twoProng0Pi0'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFindingNewDMs/hpsPFTauDiscriminationByDecayModeFindingNewDMs_pTRatio_threeProng0Pi0', 'draw': {'drawopts': 'e'}, 'description': 'QCD Jets faking taus pTRatio_threeProng0Pi0'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00ba - Fake rate from electrons vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_Matched/PFJetMatchingEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00bb - Fake rate from electrons vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00aa - Fake rate from electrons vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_Matched/PFJetMatchingEffpt', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpt', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00ab - Fake rate from electrons vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpt', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00da - Fake rate from electrons vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_Matched/PFJetMatchingEffphi', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffphi', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00db - Fake rate from electrons vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffphi', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00ca - Fake rate from electrons vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_Matched/PFJetMatchingEffeta', 'draw': {'drawopts': 'e'}, 'description': 'PFJetMatching fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffeta', 'draw': {'drawopts': 'e'}, 'description': 'DecayModeFinding fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseChargedIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00cb - Fake rate from electrons vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseIsolation/LooseIsolationEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffeta', 'draw': {'drawopts': 'e'}, 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/01b%s - Electron rejection fake rate vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'LooseElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVAElectronRejection/MVAElectronRejectionEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'MVAElectronRejection fake rate'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'MediumElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEffpileup', 'draw': {'drawopts': 'e'}, 'description': 'TightElectronRejection fake rate'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/01a - Electron rejection fake rate vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEffpt', 'draw': {'drawopts': 'e'}, 'description': 'LooseElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVA5LooseElectronRejection/MVA5LooseElectronRejectionEffpt', 'draw': {'drawopts': 'e'}, 'description': 'MVA5LooseElectronRejection fake rate'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEffpt', 'draw': {'drawopts': 'e'}, 'description': 'MediumElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEffpt', 'draw': {'drawopts': 'e'}, 'description': 'TightElectronRejection fake rate'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/01d - Electron rejection fake rate vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEffphi', 'draw': {'drawopts': 'e'}, 'description': 'LooseElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVAElectronRejection/MVAElectronRejectionEffphi', 'draw': {'drawopts': 'e'}, 'description': 'MVAElectronRejection fake rate'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEffphi', 'draw': {'drawopts': 'e'}, 'description': 'MediumElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEffphi', 'draw': {'drawopts': 'e'}, 'description': 'TightElectronRejection fake rate'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/01c - Electron rejection fake rate vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEffeta', 'draw': {'drawopts': 'e'}, 'description': 'LooseElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVAElectronRejection/MVAElectronRejectionEffeta', 'draw': {'drawopts': 'e'}, 'description': 'MVAElectronRejection fake rate'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEffeta', 'draw': {'drawopts': 'e'}, 'description': 'MediumElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEffeta', 'draw': {'drawopts': 'e'}, 'description': 'TightElectronRejection fake rate'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/04a - Distributions of size and sumPt for signalPFCands, electrons from Z faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of signalPFCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of signalPFCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_signalPFCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of signalPFCands'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/04c - Distributions of size and sumPt for isolationPFGammaCands, electrons from Z faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of isolationPFGammaCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of isolationPFGammaCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFGammaCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of isolationPFGammaCands'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/04b - Distributions of size and sumPt for isolationPFChargedHadrCands, electrons from Z faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of isolationPFChargedHadrCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of isolationPFChargedHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFChargedHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of isolationPFChargedHadrCands'}]
	)
pftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/04d - Distributions of size and sumPt for isolationPFNeutrHadrCands, electrons from Z faking taus",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_Size_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus Size of isolationPFNeutrHadrCands'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/hpsPFTauDiscriminationByDecayModeFinding_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/hpsPFTauDiscriminationByLooseChargedIsolation_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of isolationPFNeutrHadrCands'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr_SumPt_isolationPFNeutrHadrCands', 'draw': {'drawopts': 'e'}, 'description': 'electrons from Z faking taus SumPt of isolationPFNeutrHadrCands'}]
	)

apply_dqm_items_to_new_back_end(dqmitems, __file__)

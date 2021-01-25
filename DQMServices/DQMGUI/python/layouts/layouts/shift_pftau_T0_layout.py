from .adapt_to_new_backend import *
dqmitems={}

def shiftpftaulayout(i, p, *rows): i["00 Shift/Tau/" + p] = rows

shiftpftaulayout(
	dqmitems,
	"SingleMu/00a - Fake rate from muons vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpt', 'description': 'DecayModeFinding fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpt', 'description': 'LooseChargedIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpt', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpt', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpt', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpt', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/00c - Fake rate from muons vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffeta', 'description': 'DecayModeFinding fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffeta', 'description': 'LooseChargedIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffeta', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffeta', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffeta', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffeta', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/00b - Fake rate from muons vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpileup', 'description': 'DecayModeFinding fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpileup', 'description': 'LooseChargedIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpileup', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpileup', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpileup', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpileup', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/00d - Fake rate from muons vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffphi', 'description': 'DecayModeFinding fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffphi', 'description': 'LooseChargedIsolation fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffphi', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffphi', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffphi', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from muons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffphi', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from muons from Z'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/01a - Muon rejection fake rate vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEffpt', 'description': 'LooseMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEffpt', 'description': 'MediumMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEffpt', 'description': 'TightMuonRejection fake rate'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/01c - Muon rejection fake rate vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEffeta', 'description': 'LooseMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEffeta', 'description': 'MediumMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEffeta', 'description': 'TightMuonRejection fake rate'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/01b - Muon rejection fake rate vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEffpileup', 'description': 'LooseMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEffpileup', 'description': 'MediumMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEffpileup', 'description': 'TightMuonRejection fake rate'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/01d - Muon rejection fake rate vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByLooseMuonRejection/LooseMuonRejectionEffphi', 'description': 'LooseMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByMediumMuonRejection/MediumMuonRejectionEffphi', 'description': 'MediumMuonRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealMuonsData_hpsPFTauDiscriminationByTightMuonRejection/TightMuonRejectionEffphi', 'description': 'TightMuonRejection fake rate'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/02a - Fake rate from jets vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpt', 'description': 'DecayModeFinding fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpt', 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpt', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpt', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpt', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpt', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/02c - Fake rate from jets vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffeta', 'description': 'DecayModeFinding fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffeta', 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffeta', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffeta', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffeta', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffeta', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/02b - Fake rate from jets vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpileup', 'description': 'DecayModeFinding fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpileup', 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpileup', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpileup', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpileup', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpileup', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
shiftpftaulayout(
	dqmitems,
	"SingleMu/02d - Fake rate from jets vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffphi', 'description': 'DecayModeFinding fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffphi', 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffphi', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffphi', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffphi', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffphi', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
shiftpftaulayout(
	dqmitems,
	"Jet/00a - Fake rate from jets vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpt', 'description': 'DecayModeFinding fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpt', 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpt', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpt', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpt', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpt', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
shiftpftaulayout(
	dqmitems,
	"Jet/00c - Fake rate from jets vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffeta', 'description': 'DecayModeFinding fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffeta', 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffeta', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffeta', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffeta', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffeta', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
shiftpftaulayout(
	dqmitems,
	"Jet/00b - Fake rate from jets vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpileup', 'description': 'DecayModeFinding fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpileup', 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpileup', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpileup', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpileup', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpileup', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
shiftpftaulayout(
	dqmitems,
	"Jet/00d - Fake rate from jets vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffphi', 'description': 'DecayModeFinding fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffphi', 'description': 'LooseChargedIsolation fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffphi', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffphi', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffphi', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}, {'path': 'RecoTauV/hpsPFTauProducerRealData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffphi', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from QCD Jets'}]
	)
shiftpftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00a - Fake rate from electrons vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpt', 'description': 'DecayModeFinding fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpt', 'description': 'LooseChargedIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpt', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpt', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpt', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpt', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
shiftpftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00c - Fake rate from electrons vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffeta', 'description': 'DecayModeFinding fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffeta', 'description': 'LooseChargedIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffeta', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffeta', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffeta', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffeta', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
shiftpftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00b - Fake rate from electrons vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffpileup', 'description': 'DecayModeFinding fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffpileup', 'description': 'LooseChargedIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffpileup', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffpileup', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffpileup', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffpileup', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
shiftpftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/00d - Fake rate from electrons vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByDecayModeFinding/DecayModeFindingEffphi', 'description': 'DecayModeFinding fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseChargedIsolation/LooseChargedIsolationEffphi', 'description': 'LooseChargedIsolation fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr/LooseCombinedIsolationDBSumPtCorrEffphi', 'description': 'LooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr/MediumCombinedIsolationDBSumPtCorrEffphi', 'description': 'MediumCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr/TightCombinedIsolationDBSumPtCorrEffphi', 'description': 'TightCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr/VLooseCombinedIsolationDBSumPtCorrEffphi', 'description': 'VLooseCombinedIsolationDBSumPtCorr fake rate from electrons from Z'}]
	)
shiftpftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/01a - Electron rejection fake rate vs pt",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEffpt', 'description': 'LooseElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVAElectronRejection/MVAElectronRejectionEffpt', 'description': 'MVAElectronRejection fake rate'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEffpt', 'description': 'MediumElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEffpt', 'description': 'TightElectronRejection fake rate'}]
	)
shiftpftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/01c - Electron rejection fake rate vs eta",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEffeta', 'description': 'LooseElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVAElectronRejection/MVAElectronRejectionEffeta', 'description': 'MVAElectronRejection fake rate'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEffeta', 'description': 'MediumElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEffeta', 'description': 'TightElectronRejection fake rate'}]
	)
shiftpftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/01b - Electron rejection fake rate vs pileup",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEffpileup', 'description': 'LooseElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVAElectronRejection/MVAElectronRejectionEffpileup', 'description': 'MVAElectronRejection fake rate'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEffpileup', 'description': 'MediumElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEffpileup', 'description': 'TightElectronRejection fake rate'}]
	)
shiftpftaulayout(
	dqmitems,
	"DoubleElectron_OR_TauPlusX/01d - Electron rejection fake rate vs phi",
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByLooseElectronRejection/LooseElectronRejectionEffphi', 'description': 'LooseElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMVAElectronRejection/MVAElectronRejectionEffphi', 'description': 'MVAElectronRejection fake rate'}],
	[{'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByMediumElectronRejection/MediumElectronRejectionEffphi', 'description': 'MediumElectronRejection fake rate'}, {'path': 'RecoTauV/hpsPFTauProducerRealElectronsData_hpsPFTauDiscriminationByTightElectronRejection/TightElectronRejectionEffphi', 'description': 'TightElectronRejection fake rate'}]
	)


apply_dqm_items_to_new_back_end(dqmitems, __file__)

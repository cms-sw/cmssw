import FWCore.ParameterSet.Config as cms

######
# Tools to adapt Tau sequences to run tau ReReco+PAT at MiniAOD samples
# M. Bluj, NCBJ Warsaw
# based on work of J. Steggemann, CERN
# Created: 9 Nov. 2017
######

#####
def addTauReReco(process):
	#PAT
	process.load('PhysicsTools.PatAlgos.producersLayer1.tauProducer_cff')
	process.load('PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi')
	process.selectedPatTaus.cut="pt > 18. && tauID(\'decayModeFindingNewDMs\')> 0.5" #Cut as in MiniAOD
	#Tau RECO
	process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
        #Task/Sequence for tau rereco
        process.miniAODTausTask = cms.Task()
	process.miniAODTausTask.add(process.PFTauTask)
	#Add PAT Tau modules to miniAODTausTask
        process.miniAODTausTask.add(process.makePatTausTask)
        process.miniAODTausTask.add(process.selectedPatTaus)
	process.miniAODTausSequence = cms.Sequence(process.miniAODTausTask)
	#Path with tau rereco (Needed?)
        process.TauReco = cms.Path(process.miniAODTausSequence)

#####
def convertModuleToMiniAODInput(process, name):
    module = getattr(process, name)
    if hasattr(module, 'particleFlowSrc'):
        module.particleFlowSrc = cms.InputTag("packedPFCandidates", "", "")
    if hasattr(module, 'vertexSrc'):
        module.vertexSrc = cms.InputTag('offlineSlimmedPrimaryVertices')
    if hasattr(module, 'qualityCuts') and hasattr(module.qualityCuts, 'primaryVertexSrc'):
        module.qualityCuts.primaryVertexSrc = cms.InputTag('offlineSlimmedPrimaryVertices')

#####
def adaptTauToMiniAODReReco(process, reclusterJets=True):
	# TRYING TO MAKE THINGS MINIAOD COMPATIBLE, FROM THE START, TO THE END, 1 BY 1
	#print '[adaptTauToMiniAODReReco]: Start'

	jetCollection = 'slimmedJets'
	# Add new jet collections if reclustering is demanded
	if reclusterJets:
		jetCollection = 'patAK4PFJets'
		from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
		process.ak4PFJetsPAT = ak4PFJets.clone(
			src=cms.InputTag("packedPFCandidates")
		)
		# trivial PATJets
		from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import _patJets
		process.patAK4PFJets = _patJets.clone(
			jetSource            = cms.InputTag("ak4PFJetsPAT"),
			addJetCorrFactors    = cms.bool(False),
			jetCorrFactorsSource = cms.VInputTag(),
			addBTagInfo          = cms.bool(False),
			addDiscriminators    = cms.bool(False),
			discriminatorSources = cms.VInputTag(),
			addAssociatedTracks  = cms.bool(False),
			addJetCharge         = cms.bool(False),
			addGenPartonMatch    = cms.bool(False),
			embedGenPartonMatch  = cms.bool(False),
			addGenJetMatch       = cms.bool(False),
			getJetMCFlavour      = cms.bool(False),
			addJetFlavourInfo    = cms.bool(False),
		)
		process.miniAODTausTask.add(process.ak4PFJetsPAT)
		process.miniAODTausTask.add(process.patAK4PFJets)
 
	# so this adds all tracks to jet in some deltaR region. we don't have tracks so don't need it :D
	# process.ak4PFJetTracksAssociatorAtVertex.jets = cms.InputTag(jetCollection)
	
	# Remove ak4PFJetTracksAssociatorAtVertex from recoTauCommonSequence
	# Remove pfRecoTauTagInfoProducer from recoTauCommonSequence since it uses the jet-track association
	# HOWEVER, may use https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookMiniAOD2017#Isolated_Tracks
	# probably needs recovery of the two modules above


	process.recoTauAK4PatJets08Region = cms.EDProducer("RecoTauPatJetRegionProducer",
		deltaR = process.recoTauAK4PFJets08Region.deltaR,
		maxJetAbsEta = process.recoTauAK4PFJets08Region.maxJetAbsEta,
		minJetPt = process.recoTauAK4PFJets08Region.minJetPt,
		pfCandAssocMapSrc = cms.InputTag(""),
		pfCandSrc = cms.InputTag("packedPFCandidates"),
		src = cms.InputTag(jetCollection)
	)

	process.recoTauPileUpVertices.src = cms.InputTag("offlineSlimmedPrimaryVertices")
	# Redefine recoTauCommonTask 
	# with redefined region and PU vertices, and w/o track-to-vertex associator and tauTagInfo (the two latter are probably obsolete and not needed at all)
	process.recoTauCommonTask = cms.Task(
		process.recoTauAK4PatJets08Region,
		process.recoTauPileUpVertices
	)

	# Adapt TauPiZeros producer
	process.ak4PFJetsLegacyHPSPiZeros.builders[0].qualityCuts.primaryVertexSrc = cms.InputTag("offlineSlimmedPrimaryVertices")
	process.ak4PFJetsLegacyHPSPiZeros.jetSrc = cms.InputTag(jetCollection)

	# Adapt TauChargedHadrons producer
	for builder in process.ak4PFJetsRecoTauChargedHadrons.builders:
		builder.qualityCuts.primaryVertexSrc = cms.InputTag("offlineSlimmedPrimaryVertices")
		if builder.name.value() == 'tracks': #replace plugin based on generalTracks by one based on lostTracks
			builder.name = 'lostTracks'
			builder.plugin = 'PFRecoTauChargedHadronFromLostTrackPlugin'
			builder.srcLostTracks = cms.InputTag("lostTracks")
			del builder.srcTracks
	process.ak4PFJetsRecoTauChargedHadrons.jetSrc = cms.InputTag(jetCollection)

	# Adapt combinatoricRecoTau producer
	convertModuleToMiniAODInput(process, 'combinatoricRecoTaus')
	process.combinatoricRecoTaus.jetRegionSrc = 'recoTauAK4PatJets08Region'
	process.combinatoricRecoTaus.jetSrc = jetCollection
	# Adapt builders
	for builder in process.combinatoricRecoTaus.builders:
		for name,value in builder.parameters_().iteritems():
			if name == 'qualityCuts':
				builder.qualityCuts.primaryVertexSrc = 'offlineSlimmedPrimaryVertices'
			elif name == 'pfCandSrc':
				builder.pfCandSrc = 'packedPFCandidates'
	# Adapt supported modifiers and remove unsupported ones 
	modifiersToRemove_ = cms.VPSet()
	for mod in process.combinatoricRecoTaus.modifiers:
		if mod.name.value() == 'elec_rej':
			modifiersToRemove_.append(mod)
			continue
		elif mod.name.value() == 'TTIworkaround':
			modifiersToRemove_.append(mod)
			continue
		for name,value in mod.parameters_().iteritems():
			if name == 'qualityCuts':
				mod.qualityCuts.primaryVertexSrc = 'offlineSlimmedPrimaryVertices'
	for mod in modifiersToRemove_:
		process.combinatoricRecoTaus.modifiers.remove(mod)
		#print "\t\t Removing '%s' modifier from 'combinatoricRecoTaus'" %mod.name.value()

	# Adapt tau decay mode finding discrimiantor for the cleaning step
	convertModuleToMiniAODInput(process, 'hpsSelectionDiscriminator')

	# Adapt clean tau producer
	convertModuleToMiniAODInput(process, 'hpsPFTauProducerSansRefs')

	# Adapt piZero unembedder
	convertModuleToMiniAODInput(process, 'hpsPFTauProducer')

	# Adapt classic discriminants
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByDecayModeFindingNewDMs')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByDecayModeFindingOldDMs')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByDecayModeFinding')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByLooseChargedIsolation')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByLooseIsolation')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits')

	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR03')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR03')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR03')

	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits')
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits')

	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone')

	# Adapt isolation sums
	convertModuleToMiniAODInput(process, 'hpsPFTauChargedIsoPtSum')
	convertModuleToMiniAODInput(process, 'hpsPFTauNeutralIsoPtSum')
	convertModuleToMiniAODInput(process, 'hpsPFTauPUcorrPtSum')
	convertModuleToMiniAODInput(process, 'hpsPFTauNeutralIsoPtSumWeight')
	convertModuleToMiniAODInput(process, 'hpsPFTauFootprintCorrection')
	convertModuleToMiniAODInput(process, 'hpsPFTauPhotonPtSumOutsideSignalCone')
	# Adapt isolation sums (R=0.3)
	convertModuleToMiniAODInput(process, 'hpsPFTauChargedIsoPtSumdR03')
	convertModuleToMiniAODInput(process, 'hpsPFTauNeutralIsoPtSumdR03')
	convertModuleToMiniAODInput(process, 'hpsPFTauPUcorrPtSumdR03')
	convertModuleToMiniAODInput(process, 'hpsPFTauNeutralIsoPtSumWeightdR03')
	convertModuleToMiniAODInput(process, 'hpsPFTauFootprintCorrectiondR03')
	convertModuleToMiniAODInput(process, 'hpsPFTauPhotonPtSumOutsideSignalConedR03')

	# Redefine tau PV producer
	process.hpsPFTauPrimaryVertexProducer.PVTag = 'offlineSlimmedPrimaryVertices'
	process.hpsPFTauPrimaryVertexProducer.qualityCuts.primaryVertexSrc = 'offlineSlimmedPrimaryVertices'
	process.hpsPFTauPrimaryVertexProducer.packedCandidatesTag = cms.InputTag("packedPFCandidates")
	process.hpsPFTauPrimaryVertexProducer.lostCandidatesTag = cms.InputTag("lostTracks")

	# Redefine tau SV producer
	process.hpsPFTauSecondaryVertexProducer = cms.EDProducer("PFTauSecondaryVertexProducer",
		PFTauTag = cms.InputTag("hpsPFTauProducer")
	)

	# Adapt MVAIso discriminants (DBoldDMwLT)
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw')
	for wp in ['VVLoose', 'VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1DBoldDMwLT'.format(wp))
	# Adapt MVAIso discriminants (DBnewDMwLT)
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1DBnewDMwLT'.format(wp))
	# Adapt MVAIso discriminants (PWoldDMwLT)
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1PWoldDMwLT'.format(wp))
	# Adapt MVAIso discriminants (PWnewDMwLT)
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1PWnewDMwLT'.format(wp))
	# Adapt MVAIso discriminants (DBoldDMwLT, R=0.3)
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1DBdR03oldDMwLT'.format(wp))
	# Adapt MVAIso discriminants (PWoldDMwLT, R=0.3)
	convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToMiniAODInput(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1PWdR03oldDMwLT'.format(wp))

	# Remove RecoTau producers which are not supported (yet?), i.e. against-e/mu discriminats
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByTightElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByMediumElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByLooseElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByMVA6rawElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByMVA6VLooseElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByMVA6LooseElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByMVA6MediumElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByMVA6TightElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByMVA6VTightElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByDeadECALElectronRejection)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByLooseMuonRejection3)
	process.miniAODTausTask.remove(process.hpsPFTauDiscriminationByTightMuonRejection3)
	# add against-mu discriminants which are MiniAOD compatible
	process.hpsPFTauDiscriminationByLooseMuonRejectionSimple = cms.EDProducer("PFRecoTauDiscriminationAgainstMuonSimple",
		PFTauProducer = cms.InputTag("hpsPFTauProducer"),
		Prediscriminants = process.hpsPFTauDiscriminationByLooseMuonRejection3.Prediscriminants,
		HoPMin = cms.double(0.1), #use smaller value that with AOD as raw energy is used
		doCaloMuonVeto = cms.bool(False), #do not use it until tuned
		srcPatMuons = cms.InputTag("slimmedMuons"),
		minPtMatchedMuon = process.hpsPFTauDiscriminationByLooseMuonRejection3.minPtMatchedMuon,
		dRmuonMatch = process.hpsPFTauDiscriminationByLooseMuonRejection3.dRmuonMatch,
		dRmuonMatchLimitedToJetArea = process.hpsPFTauDiscriminationByLooseMuonRejection3.dRmuonMatchLimitedToJetArea,
		maskHitsCSC = process.hpsPFTauDiscriminationByLooseMuonRejection3.maskHitsCSC,
		maskHitsDT = process.hpsPFTauDiscriminationByLooseMuonRejection3.maskHitsDT,
		maskHitsRPC = process.hpsPFTauDiscriminationByLooseMuonRejection3.maskHitsRPC,
		maxNumberOfHitsLast2Stations = process.hpsPFTauDiscriminationByLooseMuonRejection3.maxNumberOfHitsLast2Stations,
		maskMatchesCSC = process.hpsPFTauDiscriminationByLooseMuonRejection3.maskMatchesCSC,
		maskMatchesDT = process.hpsPFTauDiscriminationByLooseMuonRejection3.maskMatchesDT,
		maskMatchesRPC = process.hpsPFTauDiscriminationByLooseMuonRejection3.maskMatchesRPC,
		maxNumberOfMatches = process.hpsPFTauDiscriminationByLooseMuonRejection3.maxNumberOfMatches,
		maxNumberOfSTAMuons = cms.int32(-1),
		maxNumberOfRPCMuons = cms.int32(-1),
		verbosity = cms.int32(0)
	)
	process.hpsPFTauDiscriminationByTightMuonRejectionSimple = process.hpsPFTauDiscriminationByLooseMuonRejectionSimple.clone(
		maxNumberOfHitsLast2Stations = process.hpsPFTauDiscriminationByTightMuonRejection3.maxNumberOfHitsLast2Stations
	)
	process.miniAODTausTask.add(process.hpsPFTauDiscriminationByLooseMuonRejectionSimple)
	process.miniAODTausTask.add(process.hpsPFTauDiscriminationByTightMuonRejectionSimple)

	#####
	# OK NOW COMES PATTY PAT

	# FIXME - check both if this is the OK collection...
	process.tauGenJets.GenParticles = cms.InputTag("prunedGenParticles")
	process.tauMatch.matched = cms.InputTag("prunedGenParticles")


	convertModuleToMiniAODInput(process, 'patTaus')

	# Remove unsupported tauIDs
	for name, src in process.patTaus.tauIDSources.parameters_().iteritems():
		if name.find('againstElectron') > -1 or name.find('againstMuon') > -1:
			delattr(process.patTaus.tauIDSources,name)
	# Add MiniAOD specific ones
	setattr(process.patTaus.tauIDSources,'againstMuonLooseSimple',cms.InputTag('hpsPFTauDiscriminationByLooseMuonRejectionSimple'))
	setattr(process.patTaus.tauIDSources,'againstMuonTightSimple',cms.InputTag('hpsPFTauDiscriminationByTightMuonRejectionSimple'))
	
	#print '[adaptTauToMiniAODReReco]: Done!'

#####
def setOutputModule(mode=0):
	#mode = 0: store original MiniAOD and new selectedPatTaus 
	#mode = 1: store original MiniAOD, new selectedPatTaus, and all PFTau products as in AOD (except of unsuported ones)

	import Configuration.EventContent.EventContent_cff as evtContent
	output = cms.OutputModule(
		'PoolOutputModule',
		fileName=cms.untracked.string('miniAOD_TauReco.root'),
		fastCloning=cms.untracked.bool(False),
		dataset=cms.untracked.PSet(
			dataTier=cms.untracked.string('MINIAODSIM'),
			filterName=cms.untracked.string('')
		),
		outputCommands = evtContent.MINIAODSIMEventContent.outputCommands,
		SelectEvents=cms.untracked.PSet(
			SelectEvents=cms.vstring('*',)
			)
		)
	output.outputCommands.append('keep *_selectedPatTaus_*_*')
	if mode==1:
                for prod in evtContent.RecoTauTagAOD.outputCommands:
			if prod.find('ElectronRejection') > -1:
				continue
			if prod.find('MuonRejection') > -1:
				continue
			output.outputCommands.append(prod)
		output.outputCommands.append('keep *_hpsPFTauDiscriminationByLooseMuonRejectionSimple_*_*')
		output.outputCommands.append('keep *_hpsPFTauDiscriminationByTightMuonRejectionSimple_*_*')

	return output

#####

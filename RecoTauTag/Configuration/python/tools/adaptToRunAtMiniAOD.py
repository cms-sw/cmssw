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
def convertModuleToBaseTau(process, name):
    module = getattr(process, name)
    module.__dict__['_TypedParameterizable__type'] = module.type_().replace('RecoTau', 'RecoBaseTau')
    if hasattr(module, 'PFTauProducer'):
        module.PFBaseTauProducer = module.PFTauProducer
        # del module.PFTauProducer
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
	process.ak4PFJetsRecoTauChargedHadrons.jetSrc = cms.InputTag(jetCollection)
	# FIXME - remove builder from tracks. well, because there are no tracks in miniAOD
	# One can develop similar builder from lostTracks (packedCandidates) 
	process.ak4PFJetsRecoTauChargedHadrons.builders =  cms.VPSet(process.ak4PFJetsRecoTauChargedHadrons.builders[0], process.ak4PFJetsRecoTauChargedHadrons.builders[2])

	# Adapt combinatoricRecoTau producer
	convertModuleToBaseTau(process, 'combinatoricRecoTaus')
	process.combinatoricRecoTaus.jetRegionSrc = 'recoTauAK4PatJets08Region'
	process.combinatoricRecoTaus.jetSrc = jetCollection
	# Adapt builders
	for builer in process.combinatoricRecoTaus.builders:
		builer.plugin = builer.plugin.value().replace('RecoTau', 'RecoBaseTau')
		for name,value in builer.parameters_().iteritems():
			if name == 'qualityCuts':
				builer.qualityCuts.primaryVertexSrc = 'offlineSlimmedPrimaryVertices'
			elif name == 'pfCandSrc':
				builer.pfCandSrc = 'packedPFCandidates'
	# Adapt supported modifier and remove unsupported modifers and 
	modifiersToRemove_ = cms.VPSet()
	for mod in process.combinatoricRecoTaus.modifiers:
		if mod.name.value() == 'elec_rej':
			modifiersToRemove_.append(mod)
			continue
		elif mod.name.value() == 'TTIworkaround':
			modifiersToRemove_.append(mod)
			continue
		mod.plugin = mod.plugin.value().replace('RecoTau', 'RecoBaseTau')
		for name,value in mod.parameters_().iteritems():
			if name == 'qualityCuts':
				mod.qualityCuts.primaryVertexSrc = 'offlineSlimmedPrimaryVertices'
	for mod in modifiersToRemove_:
		process.combinatoricRecoTaus.modifiers.remove(mod)
		#print "\t\t Removing '%s' modifier from 'combinatoricRecoTaus'" %mod.name.value()

	# Adapt tau decay mode finding discrimiantor for the cleaning step
	convertModuleToBaseTau(process, 'hpsSelectionDiscriminator')

	# Adapt clean tau producer
	convertModuleToBaseTau(process, 'hpsPFTauProducerSansRefs')
	# Adapt cleaners
	for cleaner in process.hpsPFTauProducerSansRefs.cleaners:
		cleaner.plugin = cleaner.plugin.value().replace('RecoTau', 'RecoBaseTau')

	# Adapt piZero unembedder
	convertModuleToBaseTau(process, 'hpsPFTauProducer')

	# Adapt classic discriminants
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByDecayModeFindingNewDMs')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByDecayModeFindingOldDMs')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByDecayModeFinding')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByLooseChargedIsolation')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByLooseIsolation')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits')

	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR03')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR03')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR03')

	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits')
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits')

	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone')

	# Adapt isolation sums
	convertModuleToBaseTau(process, 'hpsPFTauChargedIsoPtSum')
	convertModuleToBaseTau(process, 'hpsPFTauNeutralIsoPtSum')
	convertModuleToBaseTau(process, 'hpsPFTauPUcorrPtSum')
	convertModuleToBaseTau(process, 'hpsPFTauNeutralIsoPtSumWeight')
	convertModuleToBaseTau(process, 'hpsPFTauFootprintCorrection')
	convertModuleToBaseTau(process, 'hpsPFTauPhotonPtSumOutsideSignalCone')
	# Adapt isolation sums (R=0.3)
	convertModuleToBaseTau(process, 'hpsPFTauChargedIsoPtSumdR03')
	convertModuleToBaseTau(process, 'hpsPFTauNeutralIsoPtSumdR03')
	convertModuleToBaseTau(process, 'hpsPFTauPUcorrPtSumdR03')
	convertModuleToBaseTau(process, 'hpsPFTauNeutralIsoPtSumWeightdR03')
	convertModuleToBaseTau(process, 'hpsPFTauFootprintCorrectiondR03')
	convertModuleToBaseTau(process, 'hpsPFTauPhotonPtSumOutsideSignalConedR03')

	process.hpsPFTauVertexAndImpactParametersTask.remove(process.hpsPFTauPrimaryVertexProducer) #MB: Tau PV producer need be updated and added back
	# Redefine SV producer
	process.hpsPFTauSecondaryVertexProducer = cms.EDProducer("PFBaseTauSecondaryVertexProducer",
		PFTauTag = cms.InputTag("hpsPFTauProducer")
	)
	# Redefine IP producer
	process.hpsPFTauTransverseImpactParameters = cms.EDProducer("PFBaseTauTransverseImpactParameters",
		PFTauTag = cms.InputTag("hpsPFTauProducer"),
		PFTauSVATag = cms.InputTag("hpsPFTauSecondaryVertexProducer"),
		useFullCalculation = cms.bool(True),
		leadingTrkOrPFCandOption = process.combinatoricRecoTaus.builders[0].qualityCuts.leadingTrkOrPFCandOption,
		primaryVertexSrc = process.combinatoricRecoTaus.builders[0].qualityCuts.primaryVertexSrc,
		pvFindingAlgo = process.combinatoricRecoTaus.builders[0].qualityCuts.pvFindingAlgo,
		recoverLeadingTrk = process.combinatoricRecoTaus.builders[0].qualityCuts.recoverLeadingTrk,
		vxAssocQualityCuts = process.combinatoricRecoTaus.builders[0].qualityCuts.vxAssocQualityCuts,
		vertexTrackFiltering = process.combinatoricRecoTaus.builders[0].qualityCuts.vertexTrackFiltering
	)

	# Adapt MVAIso discriminants (DBoldDMwLT)
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToBaseTau(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1DBoldDMwLT'.format(wp))
	# Adapt MVAIso discriminants (DBnewDMwLT)
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToBaseTau(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1DBnewDMwLT'.format(wp))
	# Adapt MVAIso discriminants (PWoldDMwLT)
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToBaseTau(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1PWoldDMwLT'.format(wp))
	# Adapt MVAIso discriminants (PWnewDMwLT)
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToBaseTau(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1PWnewDMwLT'.format(wp))
	# Adapt MVAIso discriminants (DBoldDMwLT, R=0.3)
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToBaseTau(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1DBdR03oldDMwLT'.format(wp))
	# Adapt MVAIso discriminants (PWoldDMwLT, R=0.3)
	convertModuleToBaseTau(process, 'hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw')
	for wp in ['VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']:
		convertModuleToBaseTau(process, 'hpsPFTauDiscriminationBy{}IsolationMVArun2v1PWdR03oldDMwLT'.format(wp))

	# Remove RecoTau producers which are not supported (yet?), i.e. against-e/mu discriminats
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByTightElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByMediumElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByLooseElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByMVA6rawElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByMVA6VLooseElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByMVA6LooseElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByMVA6MediumElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByMVA6TightElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByMVA6VTightElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByDeadECALElectronRejection)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByLooseMuonRejection3)
	process.produceAndDiscriminateHPSPFTausTask.remove(process.hpsPFTauDiscriminationByTightMuonRejection3)

	#####
	# OK NOW COMES PATTY PAT

	# FIXME - check both if this is the OK collection...
	process.tauGenJets.GenParticles = cms.InputTag("prunedGenParticles")
	process.tauMatch.matched = cms.InputTag("prunedGenParticles")


	process.patTaus.__dict__['_TypedParameterizable__type'] = 'PATTauBaseProducer'
	convertModuleToBaseTau(process, 'patTaus')

	# Remove unsupported tauIDs
	for name, src in process.patTaus.tauIDSources.parameters_().iteritems():
		if name.find('againstElectron') > -1 or name.find('againstMuon') > -1:
			delattr(process.patTaus.tauIDSources,name)
	
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

	return output

#####

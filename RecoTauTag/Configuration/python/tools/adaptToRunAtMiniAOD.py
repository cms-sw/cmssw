import FWCore.ParameterSet.Config as cms
import six

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
	process.miniAODTausTask = cms.Task(
		process.PFTauTask, 
		process.makePatTausTask,
		process.selectedPatTaus
	)
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
		jetCollection = 'patJetsPAT'
		from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
		process.ak4PFJetsPAT = ak4PFJets.clone(
			src=cms.InputTag("packedPFCandidates")
		)
		# trivial PATJets
		from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import _patJets
		process.patJetsPAT = _patJets.clone(
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
		process.miniAODTausTask.add(process.patJetsPAT)
 
	# so this adds all tracks to jet in some deltaR region. we don't have tracks so don't need it :D
	# process.ak4PFJetTracksAssociatorAtVertex.jets = cms.InputTag(jetCollection)
	
	# Remove ak4PFJetTracksAssociatorAtVertex from recoTauCommonSequence
	# Remove pfRecoTauTagInfoProducer from recoTauCommonSequence since it uses the jet-track association
	# HOWEVER, may use https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookMiniAOD2017#Isolated_Tracks
	# probably needs recovery of the two modules above

	process.recoTauAK4Jets08RegionPAT = cms.EDProducer("RecoTauPatJetRegionProducer",
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
		process.recoTauAK4Jets08RegionPAT,
		process.recoTauPileUpVertices
	)

	for moduleName in process.TauReco.moduleNames(): 
		convertModuleToMiniAODInput(process, moduleName)


	# Adapt TauPiZeros producer
	process.ak4PFJetsLegacyHPSPiZeros.builders[0].qualityCuts.primaryVertexSrc = cms.InputTag("offlineSlimmedPrimaryVertices")
	process.ak4PFJetsLegacyHPSPiZeros.jetSrc = cms.InputTag(jetCollection)

	# Adapt TauChargedHadrons producer
	for builder in process.ak4PFJetsRecoTauChargedHadrons.builders:
		builder.qualityCuts.primaryVertexSrc = cms.InputTag("offlineSlimmedPrimaryVertices")
		if builder.name.value() == 'tracks': #replace plugin based on generalTracks by one based on lostTracks
			builder.name = 'lostTracks'
			builder.plugin = 'PFRecoTauChargedHadronFromLostTrackPlugin'
			builder.srcTracks = cms.InputTag("lostTracks")
	process.ak4PFJetsRecoTauChargedHadrons.jetSrc = cms.InputTag(jetCollection)

	# Adapt combinatoricRecoTau producer
	process.combinatoricRecoTaus.jetRegionSrc = 'recoTauAK4Jets08RegionPAT'
	process.combinatoricRecoTaus.jetSrc = jetCollection
	# Adapt builders
	for builder in process.combinatoricRecoTaus.builders:
		for name,value in six.iteritems(builder.parameters_()):
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
		for name,value in six.iteritems(mod.parameters_()):
			if name == 'qualityCuts':
				mod.qualityCuts.primaryVertexSrc = 'offlineSlimmedPrimaryVertices'
	for mod in modifiersToRemove_:
		process.combinatoricRecoTaus.modifiers.remove(mod)
		#print "\t\t Removing '%s' modifier from 'combinatoricRecoTaus'" %mod.name.value()

	# Redefine tau PV producer
	process.hpsPFTauPrimaryVertexProducer.__dict__['_TypedParameterizable__type'] = 'PFTauMiniAODPrimaryVertexProducer'
	process.hpsPFTauPrimaryVertexProducer.PVTag = 'offlineSlimmedPrimaryVertices'
	process.hpsPFTauPrimaryVertexProducer.packedCandidatesTag = cms.InputTag("packedPFCandidates")
	process.hpsPFTauPrimaryVertexProducer.lostCandidatesTag = cms.InputTag("lostTracks")

	# Redefine tau SV producer
	process.hpsPFTauSecondaryVertexProducer = cms.EDProducer("PFTauSecondaryVertexProducer",
		PFTauTag = cms.InputTag("hpsPFTauProducer")
	)
	
	# Remove RecoTau producers which are not supported (yet?), i.e. against-e/mu discriminats
	for moduleName in process.TauReco.moduleNames(): 
		if 'ElectronRejection' in moduleName or 'MuonRejection' in moduleName:
			process.miniAODTausTask.remove(getattr(process, moduleName))

	# Instead add against-mu discriminants which are MiniAOD compatible
	from RecoTauTag.RecoTau.hpsPFTauDiscriminationByAMuonRejectionSimple_cff import hpsPFTauDiscriminationByLooseMuonRejectionSimple, hpsPFTauDiscriminationByTightMuonRejectionSimple
	
	process.hpsPFTauDiscriminationByLooseMuonRejectionSimple = hpsPFTauDiscriminationByLooseMuonRejectionSimple
	process.hpsPFTauDiscriminationByTightMuonRejectionSimple = hpsPFTauDiscriminationByTightMuonRejectionSimple
	process.miniAODTausTask.add(process.hpsPFTauDiscriminationByLooseMuonRejectionSimple)
	process.miniAODTausTask.add(process.hpsPFTauDiscriminationByTightMuonRejectionSimple)

	#####
	# PAT part in the following

	process.tauGenJets.GenParticles = cms.InputTag("prunedGenParticles")
	process.tauMatch.matched = cms.InputTag("prunedGenParticles")

	# Remove unsupported tauIDs
	for name, src in six.iteritems(process.patTaus.tauIDSources.parameters_()):
		if name.find('againstElectron') > -1 or name.find('againstMuon') > -1:
			delattr(process.patTaus.tauIDSources,name)
	# Add MiniAOD specific ones
	setattr(process.patTaus.tauIDSources,'againstMuonLooseSimple',cms.InputTag('hpsPFTauDiscriminationByLooseMuonRejectionSimple'))
	setattr(process.patTaus.tauIDSources,'againstMuonTightSimple',cms.InputTag('hpsPFTauDiscriminationByTightMuonRejectionSimple'))
	
	#print '[adaptTauToMiniAODReReco]: Done!'

#####
def setOutputModule(mode=0):
	#mode = 0: store original MiniAOD and new selectedPatTaus 
	#mode = 1: store original MiniAOD, new selectedPatTaus, and all PFTau products as in AOD (except of unsuported ones), plus a few additional collections (charged hadrons, pi zeros, combinatoric reco taus)

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
		output.outputCommands.append('keep *_combinatoricReco*_*_*')
		output.outputCommands.append('keep *_ak4PFJetsRecoTauChargedHadrons_*_*')
		output.outputCommands.append('keep *_ak4PFJetsLegacyHPSPiZeros_*_*')
		output.outputCommands.append('keep *_patJetsPAT_*_*')

	return output

#####

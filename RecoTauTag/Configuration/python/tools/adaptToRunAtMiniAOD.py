import FWCore.ParameterSet.Config as cms

######
# Tools to adapt Tau sequences to run tau ReReco+PAT at MiniAOD samples
# M. Bluj, NCBJ Warsaw
# based on work of J. Steggemann, CERN
# Created: 9 Nov. 2017
######

import PhysicsTools.PatAlgos.tools.helpers as configtools
#####
class adaptToRunAtMiniAOD(object):
	def __init__(self, process, runBoosted=False, postfix=""):
		self.process = process
		self.runBoosted = runBoosted
		self.postfix = postfix
		if runBoosted:
			self.postfix = 'Boosted'+postfix
			#print("Adapting boosted tau reconstruction to run at miniAOD; postfix = \"%s\"" % self.postfix)
		#else:
		#	print("Adapting tau reconstruction to run at miniAOD; postfix = \"%s\"" % self.postfix)

	#####
	def addTauReReco(self):
	        #PAT
		self.process.load('PhysicsTools.PatAlgos.producersLayer1.tauProducer_cff')
		self.process.load('PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi')
		self.process.selectedPatTaus.cut="pt > 18. && tauID(\'decayModeFindingNewDMs\')> 0.5" #Cut as in MiniAOD
	        #Tau RECO
		self.process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
	        #Task/Sequence for tau rereco
		self.process.miniAODTausTask = cms.Task(
			self.process.PFTauTask,
			self.process.makePatTausTask,
			self.process.selectedPatTaus
		)
		#Add Run-2 tauIDs for boostedTaus
		if self.runBoosted:
			self.process.PFTauMVAIsoTask = cms.Task(
				self.process.hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw,
				self.process.hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT,
				self.process.hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw,
				self.process.hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT
			)
			self.process.PFTauTask.add(self.process.PFTauMVAIsoTask)
		self.miniAODTausTask = configtools.cloneProcessingSnippetTask(
			self.process,self.process.miniAODTausTask,postfix=self.postfix)
		setattr(self.process,'miniAODTausSequence'+self.postfix,cms.Sequence(self.miniAODTausTask))
		if not self.postfix=="":
			del self.process.miniAODTausTask

	#####
	def convertModuleToMiniAODInput(self,name):
		module = getattr(self.process, name)
		if hasattr(module, 'particleFlowSrc'):
			module.particleFlowSrc = cms.InputTag("packedPFCandidates", "", "")
		if hasattr(module, 'vertexSrc'):
			module.vertexSrc = cms.InputTag('offlineSlimmedPrimaryVertices')
		if hasattr(module, 'qualityCuts') and hasattr(module.qualityCuts, 'primaryVertexSrc'):
			module.qualityCuts.primaryVertexSrc = cms.InputTag('offlineSlimmedPrimaryVertices')

	#####
	def adaptTauToMiniAODReReco(self,reclusterJets=True):
	# TRYING TO MAKE THINGS MINIAOD COMPATIBLE, FROM THE START, TO THE END, 1 BY 1
	        #print '[adaptTauToMiniAODReReco]: Start'
		jetCollection = 'slimmedJets'
	        # Add new jet collections if reclustering is demanded
		if self.runBoosted:
			jetCollection = 'boostedTauSeedsPAT'+self.postfix
			from RecoTauTag.Configuration.boostedHPSPFTaus_cff import ca8PFJetsCHSprunedForBoostedTaus
			setattr(self.process,'ca8PFJetsCHSprunedForBoostedTausPAT'+self.postfix,ca8PFJetsCHSprunedForBoostedTaus.clone(
					src = 'packedPFCandidates',
					jetCollInstanceName = 'subJetsForSeedingBoostedTausPAT'
			))
			setattr(self.process,'boostedTauSeedsPAT'+self.postfix,
				cms.EDProducer("PATBoostedTauSeedsProducer",
					       subjetSrc = cms.InputTag('ca8PFJetsCHSprunedForBoostedTausPAT'+self.postfix,'subJetsForSeedingBoostedTausPAT'),
					       pfCandidateSrc = cms.InputTag('packedPFCandidates'),
					       verbosity = cms.int32(0)
			))
			self.miniAODTausTask.add(getattr(self.process,'ca8PFJetsCHSprunedForBoostedTausPAT'+self.postfix))
			self.miniAODTausTask.add(getattr(self.process,'boostedTauSeedsPAT'+self.postfix))
		elif reclusterJets:
			jetCollection = 'patJetsPAT'+self.postfix
			from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
			setattr(self.process,'ak4PFJetsPAT'+self.postfix,ak4PFJets.clone(
					src = "packedPFCandidates"
			))
		        # trivial PATJets
			from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import _patJets
			setattr(self.process,'patJetsPAT'+self.postfix,_patJets.clone(
					jetSource            = "ak4PFJetsPAT"+self.postfix,
					addJetCorrFactors    = False,
					jetCorrFactorsSource = [],
					addBTagInfo          = False,
					addDiscriminators    = False,
					discriminatorSources = [],
					addAssociatedTracks  = False,
					addJetCharge         = False,
					addGenPartonMatch    = False,
					embedGenPartonMatch  = False,
					addGenJetMatch       = False,
					getJetMCFlavour      = False,
					addJetFlavourInfo    = False,
			))
			self.miniAODTausTask.add(getattr(self.process,'ak4PFJetsPAT'+self.postfix))
			self.miniAODTausTask.add(getattr(self.process,'patJetsPAT'+self.postfix))
 
		# so this adds all tracks to jet in some deltaR region. we don't have tracks so don't need it :D
		# self.process.ak4PFJetTracksAssociatorAtVertex.jets = cms.InputTag(jetCollection)
	
		# Remove ak4PFJetTracksAssociatorAtVertex from recoTauCommonSequence
		# Remove pfRecoTauTagInfoProducer from recoTauCommonSequence since it uses the jet-track association
		# HOWEVER, may use https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookMiniAOD2017#Isolated_Tracks
		# probably needs recovery of the two modules above
		self.miniAODTausTask.remove(getattr(self.process,'ak4PFJetTracksAssociatorAtVertex'+self.postfix))
		self.miniAODTausTask.remove(getattr(self.process,'pfRecoTauTagInfoProducer'+self.postfix))

		self.miniAODTausTask.remove(getattr(self.process,'recoTauAK4PFJets08Region'+self.postfix))
		setattr(self.process,'recoTauAK4Jets08RegionPAT'+self.postfix,
			cms.EDProducer("RecoTauPatJetRegionProducer",
				       deltaR = self.process.recoTauAK4PFJets08Region.deltaR,
				       maxJetAbsEta = self.process.recoTauAK4PFJets08Region.maxJetAbsEta,
				       minJetPt = self.process.recoTauAK4PFJets08Region.minJetPt,
				       pfCandAssocMapSrc = cms.InputTag(""),
				       pfCandSrc = cms.InputTag("packedPFCandidates"),
				       src = cms.InputTag(jetCollection)
		))
		_jetRegionProducer = getattr(self.process,'recoTauAK4Jets08RegionPAT'+self.postfix)
		self.miniAODTausTask.add(_jetRegionProducer)
		if self.runBoosted:
			_jetRegionProducer.pfCandAssocMapSrc = cms.InputTag(jetCollection, 'pfCandAssocMapForIsolation')

		getattr(self.process,'recoTauPileUpVertices'+self.postfix).src = "offlineSlimmedPrimaryVertices"

		for moduleName in self.miniAODTausTask.moduleNames():
			self.convertModuleToMiniAODInput(moduleName)


		# Adapt TauPiZeros producer
		_piZeroProducer = getattr(self.process,'ak4PFJetsLegacyHPSPiZeros'+self.postfix)
		for builder in _piZeroProducer.builders:
			builder.qualityCuts.primaryVertexSrc = "offlineSlimmedPrimaryVertices"
		_piZeroProducer.jetSrc = jetCollection

		# Adapt TauChargedHadrons producer
		_chargedHadronProducer = getattr(self.process,'ak4PFJetsRecoTauChargedHadrons'+self.postfix)
		for builder in _chargedHadronProducer.builders:
			builder.qualityCuts.primaryVertexSrc = "offlineSlimmedPrimaryVertices"
			if builder.name.value() == 'tracks': #replace plugin based on generalTracks by one based on lostTracks
				builder.name = 'lostTracks'
				builder.plugin = 'PFRecoTauChargedHadronFromLostTrackPlugin'
				builder.srcTracks = "lostTracks"
				if self.runBoosted:
					builder.dRcone = 0.3
					builder.dRconeLimitedToJetArea = True
		_chargedHadronProducer.jetSrc = jetCollection

		# Adapt combinatoricRecoTau producer
		_combinatoricRecoTauProducer = getattr(self.process,'combinatoricRecoTaus'+self.postfix)
		_combinatoricRecoTauProducer.jetRegionSrc = 'recoTauAK4Jets08RegionPAT'+self.postfix
		_combinatoricRecoTauProducer.jetSrc = jetCollection
		# Adapt builders
		for builder in _combinatoricRecoTauProducer.builders:
			for name,value in builder.parameters_().items():
				if name == 'qualityCuts':
					builder.qualityCuts.primaryVertexSrc = 'offlineSlimmedPrimaryVertices'
				elif name == 'pfCandSrc':
					builder.pfCandSrc = 'packedPFCandidates'
		# Adapt supported modifiers and remove unsupported ones
		_modifiersToRemove = cms.VPSet()
		for mod in _combinatoricRecoTauProducer.modifiers:
			if mod.name.value() == 'elec_rej':
				_modifiersToRemove.append(mod)
				continue
			elif mod.name.value() == 'TTIworkaround':
				_modifiersToRemove.append(mod)
				continue
			elif mod.name.value() == 'tau_lost_tracks':
				_modifiersToRemove.append(mod)
				continue
			for name,value in mod.parameters_().items():
				if name == 'qualityCuts':
					mod.qualityCuts.primaryVertexSrc = 'offlineSlimmedPrimaryVertices'
		for mod in _modifiersToRemove:
			_combinatoricRecoTauProducer.modifiers.remove(mod)
		        #print "\t\t Removing '%s' modifier from 'combinatoricRecoTaus'" %mod.name.value()

		# Redefine tau PV producer
		_tauPVProducer =  getattr(self.process,'hpsPFTauPrimaryVertexProducer'+self.postfix)
		_tauPVProducer.__dict__['_TypedParameterizable__type'] = 'PFTauMiniAODPrimaryVertexProducer'
		_tauPVProducer.PVTag = 'offlineSlimmedPrimaryVertices'
		_tauPVProducer.packedCandidatesTag = cms.InputTag("packedPFCandidates")
		_tauPVProducer.lostCandidatesTag = cms.InputTag("lostTracks")

		# Redefine tau SV producer
		setattr(self.process,'hpsPFTauSecondaryVertexProducer'+self.postfix,
			cms.EDProducer("PFTauSecondaryVertexProducer",
				       PFTauTag = cms.InputTag("hpsPFTauProducer"+self.postfix)
		))
	
		# Remove RecoTau producers which are not supported (yet?), i.e. against-e/mu discriminats
		for moduleName in self.miniAODTausTask.moduleNames():
			if 'ElectronRejection' in moduleName or 'MuonRejection' in moduleName:
				if 'ByDeadECALElectronRejection' in moduleName: continue
				self.miniAODTausTask.remove(getattr(self.process, moduleName))

		# Instead add against-mu discriminants which are MiniAOD compatible
		from RecoTauTag.RecoTau.hpsPFTauDiscriminationByMuonRejectionSimple_cff import hpsPFTauDiscriminationByMuonRejectionSimple
	
		setattr(self.process,'hpsPFTauDiscriminationByMuonRejectionSimple'+self.postfix,
		hpsPFTauDiscriminationByMuonRejectionSimple.clone(
			PFTauProducer = "hpsPFTauProducer"+self.postfix))
		_tauIDAntiMuSimple = getattr(self.process,'hpsPFTauDiscriminationByMuonRejectionSimple'+self.postfix)
		if self.runBoosted:
			_tauIDAntiMuSimple.dRmuonMatch = 0.1
		self.miniAODTausTask.add(_tauIDAntiMuSimple)

	        #####
		# PAT part in the following

		getattr(self.process,'tauGenJets'+self.postfix).GenParticles = "prunedGenParticles"
		getattr(self.process,'tauMatch'+self.postfix).matched = "prunedGenParticles"

		# Remove unsupported tauIDs
		_patTauProducer = getattr(self.process,'patTaus'+self.postfix)
		for name,src in _patTauProducer.tauIDSources.parameters_().items():
			if name.find('againstElectron') > -1 or name.find('againstMuon') > -1:
				if name.find('againstElectronDeadECAL') > -1: continue
				delattr(_patTauProducer.tauIDSources,name)
		# Add MiniAOD specific ones
		setattr(_patTauProducer.tauIDSources,'againstMuonLooseSimple',
			cms.PSet(inputTag = cms.InputTag('hpsPFTauDiscriminationByMuonRejectionSimple'+self.postfix),
				 provenanceConfigLabel = cms.string('IDWPdefinitions'),
				 idLabel = cms.string('ByLooseMuonRejectionSimple')
                 ))
		setattr(_patTauProducer.tauIDSources,'againstMuonTightSimple',
			cms.PSet(inputTag = cms.InputTag('hpsPFTauDiscriminationByMuonRejectionSimple'+self.postfix),
				 provenanceConfigLabel = cms.string('IDWPdefinitions'),
				 idLabel = cms.string('ByTightMuonRejectionSimple')
                 ))
		#Add Run-2 tauIDs still used for boostedTaus
		if self.runBoosted:
			from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import containerID
			containerID(_patTauProducer.tauIDSources,
				"hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"+self.postfix,
				"rawValues", [
				["byIsolationMVArun2DBoldDMwLTraw", "discriminator"]
			])
			containerID(_patTauProducer.tauIDSources,
				"hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT"+self.postfix,
				"workingPoints", [
				["byVVLooseIsolationMVArun2DBoldDMwLT", "_VVLoose"],
				["byVLooseIsolationMVArun2DBoldDMwLT", "_VLoose"],
				["byLooseIsolationMVArun2DBoldDMwLT", "_Loose"],
				["byMediumIsolationMVArun2DBoldDMwLT", "_Medium"],
				["byTightIsolationMVArun2DBoldDMwLT", "_Tight"],
				["byVTightIsolationMVArun2DBoldDMwLT", "_VTight"],
				["byVVTightIsolationMVArun2DBoldDMwLT", "_VVTight"]
			])
			containerID(_patTauProducer.tauIDSources,
				"hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"+self.postfix,
				"rawValues", [
				["byIsolationMVArun2DBnewDMwLTraw", "discriminator"]
			])
			containerID(_patTauProducer.tauIDSources,
				"hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT"+self.postfix,
				"workingPoints", [
				["byVVLooseIsolationMVArun2DBnewDMwLT", "_VVLoose"],
				["byVLooseIsolationMVArun2DBnewDMwLT", "_VLoose"],
				["byLooseIsolationMVArun2DBnewDMwLT", "_Loose"],
				["byMediumIsolationMVArun2DBnewDMwLT", "_Medium"],
				["byTightIsolationMVArun2DBnewDMwLT", "_Tight"],
				["byVTightIsolationMVArun2DBnewDMwLT", "_VTight"],
				["byVVTightIsolationMVArun2DBnewDMwLT", "_VVTight"]
			])

		# Run TauIDs (anti-e && deepTau) on top of selectedPatTaus
		_updatedTauName = 'selectedPatTausNewIDs'+self.postfix
		_noUpdatedTauName = 'selectedPatTausNoNewIDs'+self.postfix
		toKeep = ['deepTau2017v2p1']
		#For boosted do not run deepTauIDs, but add still used Run-2 anti-e MVA
		if self.runBoosted:
			toKeep = ['againstEle2018']
		import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
		tauIdEmbedder = tauIdConfig.TauIDEmbedder(
			self.process, debug = False,
			originalTauName = _noUpdatedTauName,
			updatedTauName = _updatedTauName,
			postfix = "MiniAODTaus"+self.postfix,
			toKeep = toKeep
		)
		tauIdEmbedder.runTauID()
		setattr(self.process, _noUpdatedTauName, getattr(self.process,'selectedPatTaus'+self.postfix).clone())
		self.miniAODTausTask.add(getattr(self.process,_noUpdatedTauName))
		delattr(self.process, 'selectedPatTaus'+self.postfix)
		setattr(self.process,'selectedPatTaus'+self.postfix,getattr(self.process, _updatedTauName).clone())
		delattr(self.process, _updatedTauName)
		setattr(self.process,'newTauIDsTask'+self.postfix,cms.Task(
				getattr(self.process,'rerunMvaIsolationTaskMiniAODTaus'+self.postfix),
				getattr(self.process,'selectedPatTaus'+self.postfix)
		))
		self.miniAODTausTask.add(getattr(self.process,'newTauIDsTask'+self.postfix))

		#print '[adaptTauToMiniAODReReco]: Done!'

	#####
	def setOutputModule(self,mode=0):
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
		output.outputCommands.append('keep *_selectedPatTaus'+self.postfix+'_*_*')
		if mode==1:
			import re
			for prod in evtContent.RecoTauTagAOD.outputCommands:
				if prod.find('ElectronRejection') > -1 and prod.find('DeadECAL') == -1:
					continue
				if prod.find('MuonRejection') > -1:
					continue
				if prod.find("_*_*")>-1: # products w/o instance
					output.outputCommands.append(prod.replace("_*_*",self.postfix+"_*_*"))
				else: # check if there are prods with instance
					m = re.search(r'_[A-Za-z0-9]+_\*', prod)
					if m!=None:
						inst = m.group(0)
						output.outputCommands.append(prod.replace(inst,self.postfix+inst))
					else:
						print("Warning: \"%s\" do not match known name patterns; trying to keep w/o postfix" % prod)
						output.outputCommands.append(prod)

			output.outputCommands.append('keep *_hpsPFTauDiscriminationByMuonRejectionSimple'+self.postfix+'_*_*')
			output.outputCommands.append('keep *_combinatoricReco*_*_*')
			output.outputCommands.append('keep *_ak4PFJetsRecoTauChargedHadrons'+self.postfix+'_*_*')
			output.outputCommands.append('keep *_ak4PFJetsLegacyHPSPiZeros'+self.postfix+'_*_*')
			output.outputCommands.append('keep *_patJetsPAT'+self.postfix+'_*_*')
			output.outputCommands.append('keep *_boostedTauSeedsPAT'+self.postfix+'_*_*')

		return output

#####

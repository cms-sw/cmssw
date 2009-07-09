import FWCore.ParameterSet.Config as cms

def makeAODTrackCandidates(process, label='TrackCands',                ## output collection will be <'patAOD'+label>
                                tracks=cms.InputTag('generalTracks'),  ## input tracks
                                particleType="pi+",                    ## particle type (for mass)
                                candSelection='pt > 10'):              ## preselection cut on candidates
    process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi");
    setattr(process, 'patAOD' + label + 'Unfiltered', 
                     cms.EDProducer("ConcreteChargedCandidateProducer",
                        src  = tracks,
                        particleType = cms.string(particleType) ) )
    setattr(process, 'patAOD' + label,
                     cms.EDFilter("CandViewSelector",
                        src = cms.InputTag('patAOD' + label + 'Unfiltered'),
                        cut = cms.string(candSelection) ) )
    process.patAODReco.replace(process.patAODExtraReco, getattr(process, 'patAOD' + label + 'Unfiltered') * getattr(process, 'patAOD' + label)*process.patAODExtraReco)

def makePATTrackCandidates(process, 
        label='TrackCands',                     # output will be 'allLayer1'+label , 'selectedLayer1' + label
        input=cms.InputTag('patAODTrackCands'), # Name of input collection
        selection='pt > 10',                    # Selection on PAT Layer 1 objects;
                                                #   The output will be 'selectedLayer1' + label
        isolation={'tracker':0.3,               # Isolations to use ('source':deltaR)
                   'ecalTowers':0.3,            # 'tracker' => as muon track iso
                   'hcalTowers':0.3},           # 'ecalTowers', 'hcalTowers' => as muon iso from calo towers.
        isodeposits=['tracker','ecalTowers','hcalTowers'],   
        mcAs=cms.InputTag("muons") ):           # Replicate MC match as the one used by PAT on this AOD collection (None = no mc match)
    from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import allLayer1GenericParticles
    from PhysicsTools.PatAlgos.cleaningLayer1.genericTrackCleaner_cfi     import cleanLayer1Tracks
    # Define Modules
    #   producer
    setattr(process, 'allLayer1' + label, allLayer1GenericParticles.clone(src = input))
    #   selector
    setattr(process, 'selectedLayer1' + label, 
        cms.EDFilter("PATGenericParticleSelector",
            src = cms.InputTag("allLayer1"+label),
            cut = cms.string(selection) 
        ) 
    )
    #   cleaner
    setattr(process, 'cleanLayer1' + label, cleanLayer1Tracks.clone(src = cms.InputTag('selectedLayer1' + label)))
    # Get them as variables, so we can put them in the sequences and/or configure them
    l1cands = getattr(process, 'allLayer1' + label)
    selectedL1cands = getattr(process, 'selectedLayer1' + label)
    cleanL1cands    = getattr(process, 'cleanLayer1' + label)
    # Insert in sequence, after electrons
    process.allLayer1Objects.replace(process.allLayer1Electrons, l1cands + process.allLayer1Electrons)
    process.selectedLayer1Objects.replace(process.selectedLayer1Electrons, process.selectedLayer1Electrons + selectedL1cands)
    process.cleanLayer1Objects.replace(process.cleanLayer1Electrons, process.cleanLayer1Electrons + cleanL1cands)
    # Add to Summary Tables
    process.aodSummary.candidates += [ input ]
    process.allLayer1Summary.candidates += [ cms.InputTag("allLayer1"+label) ]
    process.selectedLayer1Summary.candidates += [ cms.InputTag("selectedLayer1"+label) ]
    process.cleanLayer1Summary.candidates += [ cms.InputTag("cleanLayer1"+label) ]
    
    # Isolation: start with empty config
    if isolation or isodeposits:
        process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
        process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
        process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
    isoModules = []
    runIsoDeps = { 'tracker':False, 'caloTowers':False }
    for (source,deltaR) in isolation.items():
        if source == 'tracker':
            runIsoDeps['tracker'] = True
            l1cands.isolation.tracker = cms.PSet(
                    src    = cms.InputTag('pat'+label+'IsoDepositTracks'),
                    deltaR = cms.double(deltaR),
            )
        elif source == 'ecalTowers':
            runIsoDeps['caloTowers'] = True
            l1cands.isolation.ecal = cms.PSet(
                    src    = cms.InputTag('pat'+label+'IsoDepositCaloTowers', 'ecal'),
                    deltaR = cms.double(deltaR),
            )
        elif source == 'hcalTowers':
            runIsoDeps['caloTowers'] = True
            l1cands.isolation.hcal = cms.PSet(
                    src    = cms.InputTag('pat'+label+'IsoDepositCaloTowers', 'hcal'),
                    deltaR = cms.double(deltaR),
            )
    for source in isodeposits:
        if source == 'tracker':
            runIsoDeps['tracker'] = True
            l1cands.isoDeposits.tracker = cms.InputTag('pat'+label+'IsoDepositTracks') 
        elif source == 'ecalTowers':
            runIsoDeps['caloTowers'] = True
            l1cands.isoDeposits.ecal = cms.InputTag('pat'+label+'IsoDepositCaloTowers', 'ecal') 
        elif source == 'hcalTowers':
            runIsoDeps['caloTowers'] = True
            l1cands.isoDeposits.hcal = cms.InputTag('pat'+label+'IsoDepositCaloTowers', 'hcal') 
    for dep in [ dep for dep,runme in runIsoDeps.items() if runme == True ]:
        if dep == 'tracker':
            from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import MIsoTrackExtractorCtfBlock
            setattr(process, 'pat'+label+'IsoDepositTracks',
                             cms.EDProducer("CandIsoDepositProducer",
                                    src                  = input,
                                    trackType            = cms.string('best'),
                                    MultipleDepositsFlag = cms.bool(False),
                                    ExtractorPSet        = cms.PSet( MIsoTrackExtractorCtfBlock )
                                ) )
            isoModules.append( getattr(process, 'pat'+label+'IsoDepositTracks') )
        elif dep == 'caloTowers':
            from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import MIsoCaloExtractorByAssociatorTowersBlock
            setattr(process, 'pat'+label+'IsoDepositCaloTowers',
                             cms.EDProducer("CandIsoDepositProducer",
                                    src                  = input,
                                    trackType            = cms.string('best'),
                                    MultipleDepositsFlag = cms.bool(True),
                                    ExtractorPSet        = cms.PSet( MIsoCaloExtractorByAssociatorTowersBlock )
                                ) )
            isoModules.append( getattr(process, 'pat'+label+'IsoDepositCaloTowers') )
    for m in isoModules: process.patAODExtraReco += m
    # MC 
    from PhysicsTools.PatAlgos.tools.jetTools import MassSearchParamVisitor;
    if type(mcAs) != type(None): # otherwise it complains that 'None' is not a valid InputTag :-(
        searchMC = MassSearchParamVisitor('src', mcAs);
        process.patMCTruth.visit(searchMC)
        modulesMC = searchMC.modules()
        if len(modulesMC) != 1: raise RuntimeError, "Can't find MC-Truth match for '%s', or it's not unique."%(mcAs,)
        setattr(process, 'pat'+label+'MCMatch', modulesMC[0].clone(src = input))
        process.patMCTruth.replace( modulesMC[0], modulesMC[0] + getattr(process, 'pat'+label+'MCMatch'))
        l1cands.addGenMatch = True
        l1cands.genParticleMatch = cms.InputTag('pat'+label+'MCMatch')

def makeTrackCandidates(process, 
        label='TrackCands',                   # output collection will be 'allLayer'+(0/1)+label , 'selectedLayer1' + label
        tracks=cms.InputTag('generalTracks'), # input track collection
        particleType="pi+",                   # particle type (for assigning a mass)
        preselection='pt > 10',               # preselection cut on candidates
        selection='pt > 10',                  # Selection on PAT Layer 1 objects. The output will be 'selectedLayer1' + label
        isolation={'tracker':   0.3,          #  Isolations to use ({'source':deltaR, ...}; use {} for none)
                   'ecalTowers':0.3,          # 'tracker' => as muon track iso
                   'hcalTowers':0.3},         # 'ecalTowers', 'hcalTowers' => as muon iso from calo towers.
        isodeposits=['tracker','ecalTowers','hcalTowers'],   # IsoDeposits to save ([] for none)
        mcAs=cms.InputTag("muons") ) :        # Replicate MC match as the one used by PAT on this AOD collection (None = no mc match)
    makeAODTrackCandidates(process, tracks=tracks, particleType=particleType, candSelection=preselection, label=label) 
    makePATTrackCandidates(process, label=label, input=cms.InputTag('patAOD' + label), 
                           isolation=isolation, isodeposits=isodeposits, mcAs=mcAs, selection=selection)

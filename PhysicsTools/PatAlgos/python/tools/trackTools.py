import FWCore.ParameterSet.Config as cms

def makeAODTrackCandidates(process,
                           label         = 'TrackCands',                 
                           tracks        = cms.InputTag('generalTracks'),
                           particleType  = "pi+",                        
                           candSelection = 'pt > 10'                     
                           ):
    """
    ------------------------------------------------------------------
    create selected tracks and a candidate hypothesis on AOD:
    
    process       : process
    label         : output collection will be <'patAOD'+label>
    tracks        : input tracks
    particleType  : particle type (for mass) 
    candSelection : preselection cut on the candidates
    ------------------------------------------------------------------    
    """
    process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi");
    ## add ChargedCandidateProducer from track
    setattr(process, 'patAOD' + label + 'Unfiltered', cms.EDProducer("ConcreteChargedCandidateProducer",
                                                                     src  = tracks,
                                                                     particleType = cms.string(particleType)
                                                                     )
            )
    ## add CandViewSelector with preselection string
    setattr(process, 'patAOD' + label, cms.EDFilter("CandViewSelector",
                                                    src = cms.InputTag('patAOD' + label + 'Unfiltered'),
                                                    cut = cms.string(candSelection)
                                                    )
            )
    ## run production of TrackCandidates at the very beginning of the sequence
    process.patDefaultSequence.replace(process.patCandidates, getattr(process, 'patAOD' + label + 'Unfiltered') * getattr(process, 'patAOD' + label) * process.patCandidates)

    
def makePATTrackCandidates(process, 
                           label       = 'TrackCands',                    
                           input       = cms.InputTag('patAODTrackCands'),
                           selection   = 'pt > 10',                       
                           isolation   = {'tracker':0.3, 'ecalTowers':0.3, 'hcalTowers':0.3},  
                           isoDeposits = ['tracker','ecalTowers','hcalTowers'],   
                           mcAs        = 'muon'            
                           ):
    """
    ------------------------------------------------------------------
    create pat track candidates from AOD track collections:
    
    process       : process
    label         : output will be 'all/selectedPat'+label
    input         : name of the input collection
    selection     : selection on PAT Layer 1 objects
    isolation     : isolation to use (as 'source': value of dR)
                    tracker     : as muon iso from tracks
                    ecalTowers  : as muon iso from calo towers.
                    hcalTowers  : as muon iso from calo towers.
    isoDeposits   : iso deposits
    mcAs          : replicate mc match as the one used by PAT
                    on this AOD collection (None=no mc match);
                    chosse 'photon', 'electron', 'muon', 'tau',
                    'jet', 'met' as input string
    ------------------------------------------------------------------    
    """
    ## add patTracks to the process
    from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import patGenericParticles
    setattr(process, 'pat' + label, patGenericParticles.clone(src = input))
    ## add selectedPatTracks to the process
    setattr(process, 'selectedPat' + label, cms.EDFilter("PATGenericParticleSelector",
                                                            src = cms.InputTag("pat"+label),
                                                            cut = cms.string(selection) 
                                                            ) 
            )
    ## add cleanPatTracks to the process
    from PhysicsTools.PatAlgos.cleaningLayer1.genericTrackCleaner_cfi import cleanPatTracks
    setattr(process, 'cleanPat' + label, cleanPatTracks.clone(src = cms.InputTag('selectedPat' + label)))

    ## get them as variables, so we can put them in the sequences and/or configure them
    l1cands         = getattr(process, 'pat' + label)
    selectedL1cands = getattr(process, 'selectedPat' + label)
    cleanL1cands    = getattr(process, 'cleanPat' + label)

    ## insert them in sequence, after the electrons
    process.patCandidates.replace(process.patElectrons, l1cands + process.patElectrons)
    process.selectedPatCandidates.replace(process.selectedPatElectrons, process.selectedPatElectrons + selectedL1cands)
    process.cleanPatCandidates.replace(process.cleanPatElectrons, process.cleanPatElectrons + cleanL1cands)
    
    ## add them to the Summary Tables
    process.patCandidateSummary.candidates += [ cms.InputTag("allPat"+label) ]
    process.selectedPatCandidateSummary.candidates += [ cms.InputTag("selectedPat"+label) ]
    process.cleanPatCandidateSummary.candidates += [ cms.InputTag("cleanPat"+label) ]
    
    ## isolation: start with empty config
    if(isolation or isoDeposits):
        process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
        process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
        process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
    isoModules = []
    runIsoDeps = {'tracker':False, 'caloTowers':False}
    
    for source,deltaR in isolation.items():
        ## loop items in isolation
        if(source == 'tracker'):
            runIsoDeps['tracker'] = True
            l1cands.userIsolation.tracker = cms.PSet(
                    src    = cms.InputTag('pat'+label+'IsoDepositTracks'),
                    deltaR = cms.double(deltaR),
            )
        elif(source == 'ecalTowers'):
            runIsoDeps['caloTowers'] = True
            l1cands.userIsolation.ecal = cms.PSet(
                    src    = cms.InputTag('pat'+label+'IsoDepositCaloTowers', 'ecal'),
                    deltaR = cms.double(deltaR),
            )
        elif(source == 'hcalTowers'):
            runIsoDeps['caloTowers'] = True
            l1cands.userIsolation.hcal = cms.PSet(
                    src    = cms.InputTag('pat'+label+'IsoDepositCaloTowers', 'hcal'),
                    deltaR = cms.double(deltaR),
            )
            
    for source in isoDeposits:
        ## loop items in isoDeposits
        if(source == 'tracker'):
            runIsoDeps['tracker'] = True
            l1cands.isoDeposits.tracker = cms.InputTag('pat'+label+'IsoDepositTracks') 
        elif(source == 'ecalTowers'):
            runIsoDeps['caloTowers'] = True
            l1cands.isoDeposits.ecal = cms.InputTag('pat'+label+'IsoDepositCaloTowers', 'ecal') 
        elif(source == 'hcalTowers'):
            runIsoDeps['caloTowers'] = True
            l1cands.isoDeposits.hcal = cms.InputTag('pat'+label+'IsoDepositCaloTowers', 'hcal')
            
    for dep in [ dep for dep,runme in runIsoDeps.items() if runme == True ]:
        if(dep == 'tracker'):
            from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import MIsoTrackExtractorCtfBlock
            setattr(process, 'pat'+label+'IsoDepositTracks',
                    cms.EDProducer("CandIsoDepositProducer",
                                   src                  = input,
                                   trackType            = cms.string('best'),
                                   MultipleDepositsFlag = cms.bool(False),
                                   ExtractorPSet        = cms.PSet( MIsoTrackExtractorCtfBlock )
                                   )
                    )
            isoModules.append( getattr(process, 'pat'+label+'IsoDepositTracks') )
        elif(dep == 'caloTowers'):
            from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import MIsoCaloExtractorByAssociatorTowersBlock
            setattr(process, 'pat'+label+'IsoDepositCaloTowers',
                    cms.EDProducer("CandIsoDepositProducer",
                                   src                  = input,
                                   trackType            = cms.string('best'),
                                   MultipleDepositsFlag = cms.bool(True),
                                   ExtractorPSet        = cms.PSet( MIsoCaloExtractorByAssociatorTowersBlock )
                                   )
                    )
            isoModules.append( getattr(process, 'pat'+label+'IsoDepositCaloTowers') )
    for m in isoModules:
        process.patDefaultSequence.replace(l1cands, m * l1cands)
        
    # MC
    from PhysicsTools.PatAlgos.tools.helpers import MassSearchParamVisitor
    if(type(mcAs) != type(None)):
        findMatch= []
        findMatch.append(getattr(process, mcAs+'Match'))

        ## clone mc matchiong module of object mcAs and add it to the path
        setattr(process, 'pat'+label+'MCMatch', findMatch[0].clone(src = input))
        process.patDefaultSequence.replace( l1cands, getattr(process, 'pat'+label+'MCMatch') * l1cands)
        l1cands.addGenMatch = True
        l1cands.genParticleMatch = cms.InputTag('pat'+label+'MCMatch')


def makeTrackCandidates(process, 
                        label        = 'TrackCands',                 
                        tracks       = cms.InputTag('generalTracks'),
                        particleType = 'pi+',                        
                        preselection = 'pt > 10',                    
                        selection    = 'pt > 10',                     
                        isolation    = {'tracker':   0.3,             
                                        'ecalTowers':0.3,             
                                        'hcalTowers':0.3              
                                        },
                        isoDeposits  = ['tracker','ecalTowers','hcalTowers'],
                        mcAs         = 'muon'
                        ) :
    """
    ------------------------------------------------------------------
    create selected tracks and a candidate hypothesis on AOD:
    
    process       : process
    label         : output collection will be <'patAOD'+label>
    tracks        : input tracks
    particleType  : particle type (for mass) 
    preselection  : preselection cut on the AOD candidates
    selection     : selection cut on the PAT candidates (for the
                    selectedPatCandidate collection)
    isolation     : isolation to use (as 'source': value of dR)
                    tracker     : as muon iso from tracks
                    ecalTowers  : as muon iso from calo towers.
                    hcalTowers  : as muon iso from calo towers.
    isoDeposits   : iso deposits
    mcAs          : replicate mc match as the one used by PAT
                    on this AOD collection (None=no mc match);
                    chosse 'photon', 'electron', 'muon', 'tau',
                    'jet', 'met' as input string
    ------------------------------------------------------------------    
    """    
    makeAODTrackCandidates(process,
                           tracks        = tracks,
                           particleType  = particleType,
                           candSelection = preselection,
                           label         = label
                           ) 
    makePATTrackCandidates(process,
                           label         = label,
                           input         = cms.InputTag('patAOD' + label), 
                           isolation     = isolation,
                           isoDeposits   = isoDeposits,
                           mcAs          = mcAs,
                           selection     = selection
                           )

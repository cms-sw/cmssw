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
    process.patLayer0.replace( process.patBeforeLevel0Reco, 
            getattr(process, 'patAOD' + label + 'Unfiltered')
            * getattr(process, 'patAOD' + label)
            + process.patBeforeLevel0Reco )

def makePATTrackCandidates(process, label='TrackCands',                   # output will be 'allLayer'+(0/1)+label , 'selectedLayer1' + label
                                    input='patAODTrackCands',             # Name of input collection
                                    cleaning=True,                        # Run the 'PATGenericParticleCleaner'
                                                                          #   The module will be called 'allLayer0' + label
                                    selection='pt > 10',                  # Selection on PAT Layer 1 objects;
                                                                          #   The output will be 'selectedLayer1' + label
                                                                          #   If set to None, you will only have 'allLayer1'+label in the output file
                                    isolation=['tracker','caloTowers'],   # Isolations to use (currently only 'tracker' and 'caloTowers' are valid options)
                                    mcAs='allLayer0Muons',                # Replicate MC match as the one used for this PAT collection (None = no mc match)
                                    triggerAs=['allLayer0Muons'],         # Replicate trigger match as all the ones used by these PAT collections (None = no trig.)
                                    layers=[0,1]):                        # Apply to PAT Layer 0 and 1  (the only other valid option is [0], not also [1])
    l1cands = None; l0cands = None; l0label = None;
    if cleaning :
        from PhysicsTools.PatAlgos.cleaningLayer0.genericTrackCleaner_cfi import allLayer0TrackCands
        l0label = 'allLayer0' + label
        setattr(process, l0label, allLayer0TrackCands.clone(src = cms.InputTag(input)))
        l0cands = getattr(process, l0label)
        process.patLayer0.replace(process.allLayer0Jets, l0cands + process.allLayer0Jets)
    else:
        l0label = Input
    if layers.count(1) != 0:
        from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import allLayer1GenericParticles
        setattr(process, 'allLayer1' + label, allLayer1GenericParticles.clone(src = cms.InputTag(l0label)))
        l1cands = getattr(process, 'allLayer1' + label)
        process.patLayer1.replace(process.allLayer1Jets, l1cands + process.allLayer1Jets)
        if selection != None:
            setattr(process, 'selectedLayer1' + label, 
                             cms.EDFilter("PATGenericParticleSelector",
                                src = cms.InputTag("allLayer1"+label),
                                cut = cms.string(selection) ) )
            process.patLayer1.replace(process.selectedLayer1Jets, 
                                        getattr(process, 'selectedLayer1' + label) + process.selectedLayer1Jets)
    
    # Isolation: start with empty config
    if isolation == None or isolation == []:
        if l0cands != None:
            l0cands.isolation = cms.PSet()
        if l1cands != None:
            l1cands.isolation = cms.PSet()
            l1cands.isoDeposits = cms.PSet()
    else:
        isoLabels = []; isoModules = []
        isoDets = { 'tracker':False, 'ecal':False, 'hcal':False }
        for source in isolation:
            if source == 'tracker':
                isoDets['tracker'] = True
                from RecoMuon.MuonIsolationProducers.trackExtractorBlocks_cff import MIsoTrackExtractorCtfBlock
                setattr(process, 'patAOD'+label+'IsoDepositTracks',
                                 cms.EDProducer("CandIsoDepositProducer",
                                        src                  = cms.InputTag(input),
                                        trackType            = cms.string('best'),
                                        MultipleDepositsFlag = cms.bool(False),
                                        ExtractorPSet        = cms.PSet( MIsoTrackExtractorCtfBlock )
                                    ) )
                isoLabels.append(cms.InputTag('patAOD'+label+'IsoDepositTracks'))
                isoModules.append(getattr(process, 'patAOD'+label+'IsoDepositTracks'))
                if l0cands != None:
                    l0cands.isolation.tracker.src = cms.InputTag('patAOD'+label+'Isolations', 'patAOD'+label+'IsoDepositTracks') 
                if l1cands != None: 
                    l1cands.isolation.tracker.src = cms.InputTag('layer0'+label+'Isolations', 'patAOD'+label+'IsoDepositTracks') 
                    l1cands.isoDeposits.tracker   = cms.InputTag('layer0'+label+'Isolations', 'patAOD'+label+'IsoDepositTracks') 
            elif source == 'caloTowers':
                isoDets['ecal'] = True
                isoDets['hcal'] = True
                from RecoMuon.MuonIsolationProducers.caloExtractorByAssociatorBlocks_cff import MIsoCaloExtractorByAssociatorTowersBlock
                setattr(process, 'patAOD'+label+'IsoDepositCaloTowers',
                                 cms.EDProducer("CandIsoDepositProducer",
                                        src                  = cms.InputTag(input),
                                        trackType            = cms.string('best'),
                                        MultipleDepositsFlag = cms.bool(True),
                                        ExtractorPSet        = cms.PSet( MIsoCaloExtractorByAssociatorTowersBlock )
                                    ) )
                isoLabels += [ cms.InputTag('patAOD'+label+'IsoDepositCaloTowers',x) for x in ('ecal', 'hcal') ]
                isoModules.append(getattr(process, 'patAOD'+label+'IsoDepositCaloTowers') )
                if l0cands != None:
                    l0cands.isolation.ecal.src = cms.InputTag('patAOD'+label+'Isolations', 'patAOD'+label+'IsoDepositCaloTowers' + 'ecal')
                    l0cands.isolation.hcal.src = cms.InputTag('patAOD'+label+'Isolations', 'patAOD'+label+'IsoDepositCaloTowers' + 'hcal')
                if l1cands != None:
                    l1cands.isolation.ecal.src   = cms.InputTag('layer0'+label+'Isolations', 'patAOD'+label+'IsoDepositCaloTowers' + 'ecal')
                    l1cands.isolation.hcal.src   = cms.InputTag('layer0'+label+'Isolations', 'patAOD'+label+'IsoDepositCaloTowers' + 'hcal')
                    l1cands.isoDeposits.ecal = cms.InputTag('layer0'+label+'Isolations', 'patAOD'+label+'IsoDepositCaloTowers' + 'ecal')
                    l1cands.isoDeposits.hcal = cms.InputTag('layer0'+label+'Isolations', 'patAOD'+label+'IsoDepositCaloTowers' + 'hcal')
            else:
                raise ValueError, "Unknown isolation '%s'"%(source,)
        # turn off unused dets
        for det in [ x for (x,y) in isoDets.items() if y == False]: # loop on unread detector labels
            if l0cands != None:
                setattr(l0cands.isolation, source, cms.PSet())
            if l1cands != None:
                setattr(l1cands.isolation,   source, cms.PSet())
                setattr(l1cands.isoDeposits, source, cms.InputTag(''))
        # now make up sequences, converter, re-keyer
        seq = isoModules[0];
        for m in isoModules[1:]: seq += m
        if l0cands != None:
            setattr(process, 'patAOD'+label+'Isolations', 
                             cms.EDFilter("MultipleIsoDepositsToValueMaps",
                                collection   = cms.InputTag(input),
                                associations = cms.VInputTag(*isoLabels) ) )
            seq += getattr(process, 'patAOD'+label+'Isolations')
            setattr(process, 'layer0'+label+'Isolations',
                             cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
                                 collection   = cms.InputTag(l0label),
                                 backrefs     = cms.InputTag(l0label),
                                 commonLabel  = cms.InputTag('patAOD'+label+'Isolations'),
                                 associations = cms.VInputTag(*isoLabels) ) )
            setattr(process, 'patAOD'+label+'Isolation', cms.Sequence( seq ) )
            # put AOD iso
            process.patLayer0.replace(process.patBeforeLevel0Reco, process.patBeforeLevel0Reco + getattr(process, 'patAOD'+label+'Isolation'))
            # put Keyeyer
            process.patLayer0.replace(process.patHighLevelReco,    process.patHighLevelReco    + getattr(process, 'layer0'+label+'Isolations'))  
        else:
            setattr(process, 'layer0'+label+'Isolations', 
                             cms.EDFilter("MultipleIsoDepositsToValueMaps",
                                collection   = cms.InputTag(input),
                                associations = cms.VInputTag(*isoLabels) ) )
            # do directly in post-layer0
            setattr(process, 'layer0'+label+'Isolation', cms.Sequence( seq ) )
            process.patLayer0.replace(process.patHighLevelReco,    process.patHighLevelReco    + getattr(process, 'layer0'+label+'Isolation'))  
    
    # MC and trigger
    from PhysicsTools.PatAlgos.tools.jetTools import MassSearchParamVisitor;
    if mcAs != None:
        searchMC = MassSearchParamVisitor('src', cms.InputTag(mcAs));
        process.patMCTruth.visit(searchMC)
        modulesMC = searchMC.modules()
        if len(modulesMC) != 1: raise RuntimeError, "Can't find MC-Truth match for '%s', or it's not unique."%(mcAs,)
        setattr(process, 'pat'+label+'MCMatch', modulesMC[0].clone(src = cms.InputTag(l0label)))
        process.patMCTruth.replace( modulesMC[0], modulesMC[0] + getattr(process, 'pat'+label+'MCMatch'))
        if l1cands != None:
            l1cands.addGenMatch = True
            l1cands.genParticleMatch = cms.InputTag('pat'+label+'MCMatch')
    if triggerAs != None and triggerAs != []:
        modulesTR = []; labelsTR = []
        for t in triggerAs:
            searchTR = MassSearchParamVisitor('src', cms.InputTag(t));
            process.patTrigMatch.visit(searchTR)
            modulesTR += searchTR.modules()
        if len(modulesTR) == 0: raise RuntimeError, "Can't find any trigger match among %s" % (triggerAs)
        def ucfirst(x): return x[0].upper() + x[1:]
        for m in modulesTR:
            lbl = 'pat'+label+'TrigMatchAs' + ucfirst(m.label())
            setattr(process, lbl, m.clone(src = cms.InputTag(l0label)))
            process.patTrigMatch.replace( m, m + getattr(process, lbl))
            labelsTR.append (cms.InputTag(lbl))
        if l1cands != None:
            l1cands.addTrigMatch = cms.bool(True)
            l1cands.trigPrimMatch = cms.VInputTag(*labelsTR)

def makeTrackCandidates(process, 
        label='TrackCands',                   # output collection will be 'allLayer'+(0/1)+label , 'selectedLayer1' + label
        tracks=cms.InputTag('generalTracks'), # input track collection
        particleType="pi+",                   # particle type (for assigning a mass)
        preselection='pt > 10',               # preselection cut on candidates
        cleaning=True,                        # Run the PATGenericParticleCleaner 
                                              #   The module will be called 'allLayer0' + label
        selection='pt > 10',                  # Selection on PAT Layer 1 objects; 
                                              #   The output will be 'selectedLayer1' + label
                                              #   If set to None, you will only have 'allLayer1'+label in the output file
        isolation=['tracker','caloTowers'],   # Isolations to use (currently only 'tracker' and 'caloTowers' are valid options)
        mcAs='allLayer0Muons',                # Replicate MC match as the one used for this PAT collection (None = no mc match)
        triggerAs=['allLayer0Muons'],         # Replicate trigger match as all the ones used by these PAT collections (None = no trig.)
        layers=[0,1]):
    makeAODTrackCandidates(process, tracks=tracks, particleType=particleType, candSelection=preselection, label=label) 
    makePATTrackCandidates(process, label=label, input='patAOD' + label, cleaning=cleaning, selection=selection, isolation=isolation, mcAs=mcAs, triggerAs=triggerAs, layers=layers)

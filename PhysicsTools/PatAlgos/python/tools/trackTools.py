from FWCore.GuiBrowsers.ConfigToolBase import *


class MakeAODTrackCandidates(ConfigToolBase):

    """ Create selected tracks and a candidate hypothesis on AOD:
    """
    _label='makeAODTrackCandidates'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'label','TrackCands', "output collection will be <'patAOD'+label>")
        self.addParameter(self._defaultParameters,'tracks',cms.InputTag('generalTracks'), 'input tracks')
        self.addParameter(self._defaultParameters,'particleType','pi+', 'particle type (for mass)')
        self.addParameter(self._defaultParameters,'candSelection','pt > 10', 'preselection cut on the candidates')

        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 label         = None,
                 tracks        = None,
                 particleType  = None,
                 candSelection = None) :
        if label  is None:
            label=self._defaultParameters['label'].value
        if  tracks is None:
            tracks=self._defaultParameters['tracks'].value
        if  particleType is None:
            particleType=self._defaultParameters['particleType'].value
        if  candSelection is None:
            candSelection=self._defaultParameters['candSelection'].value
        self.setParameter('label',label)
        self.setParameter('tracks',tracks)
        self.setParameter('particleType',particleType)
        self.setParameter('candSelection',candSelection)
        self.apply(process)

    def toolCode(self, process):
        label=self._parameters['label'].value
        tracks=self._parameters['tracks'].value
        particleType=self._parameters['particleType'].value
        candSelection=self._parameters['candSelection'].value

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

makeAODTrackCandidates=MakeAODTrackCandidates()


class MakePATTrackCandidates(ConfigToolBase):

    """ Create pat track candidates from AOD track collections:
    """
    _label='makePATTrackCandidates'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'label','TrackCands', "output will be 'all/selectedLayer1'+label")
        self.addParameter(self._defaultParameters,'input',cms.InputTag('patAODTrackCands'), 'name of the input collection')
        self.addParameter(self._defaultParameters,'selection','pt > 10', 'selection on PAT Layer 1 objects')
        self.addParameter(self._defaultParameters,'isolation',{'tracker':0.3, 'ecalTowers':0.3, 'hcalTowers':0.3}, "solation to use (as 'source': value of dR)\ntracker     : as muon iso from tracks\necalTowers  : as muon iso from calo tower\nhcalTowers  : as muon iso from calo towers",allowedValues=['tracker','ecalTowers','hcalTowers'])
        self.addParameter(self._defaultParameters,'isoDeposits',['tracker','ecalTowers','hcalTowers'], 'iso deposits')
        self.addParameter(self._defaultParameters,'mcAs',None, "eplicate mc match as the one used by PAT on this AOD collection (None=no mc match); choose 'photon', 'electron', 'muon', 'tau','jet', 'met' as input string",Type=str, allowedValues=['photon', 'electron', 'muon', 'tau','jet', 'met', None], acceptNoneValue = True)

        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 label       = None,
                 input       = None,
                 selection   = None,
                 isolation   = None,
                 isoDeposits = None,
                 mcAs        = None) :
        if label  is None:
            label=self._defaultParameters['label'].value
        if input is None:
            input=self._defaultParameters['input'].value
        if selection is None:
            selection=self._defaultParameters['selection'].value
        if isolation is None:
            isolation=self._defaultParameters['isolation'].value
        if isoDeposits is None:
            isoDeposits=self._defaultParameters['isoDeposits'].value
        if mcAs is None:
            mcAs=self._defaultParameters['mcAs'].value
        self.setParameter('label',label)
        self.setParameter('input',input)
        self.setParameter('selection',selection)
        self.setParameter('isolation',isolation)
        self.setParameter('isoDeposits',isoDeposits)
        self.setParameter('mcAs',mcAs,True)
        self.apply(process)

    def toolCode(self, process):
        label=self._parameters['label'].value
        input=self._parameters['input'].value
        selection=self._parameters['selection'].value
        isolation=self._parameters['isolation'].value
        isoDeposits=self._parameters['isoDeposits'].value
        mcAs=self._parameters['mcAs'].value

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

        ### add them to the Summary Tables
        #process.patCandidateSummary.candidates += [ cms.InputTag("allPat"+label) ]
        #process.selectedPatCandidateSummary.candidates += [ cms.InputTag("selectedPat"+label) ]
        #process.cleanPatCandidateSummary.candidates += [ cms.InputTag("cleanPat"+label) ]

        ## isolation: start with empty config
        if(isolation or isoDeposits):
            process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
            process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
            process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
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
        # ES
        process.load( 'TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff' )
        # MC
        from PhysicsTools.PatAlgos.tools.helpers import MassSearchParamVisitor
        if(type(mcAs) != type(None)):
            findMatch= []
            findMatch.append(getattr(process, mcAs+'Match'))

            ## clone mc matchiong module of object mcAs and add it to the path
            setattr(process, 'pat'+label+'MCMatch', findMatch[0].clone(src = input))
            l1cands.addGenMatch = True
            l1cands.genParticleMatch = cms.InputTag('pat'+label+'MCMatch')


makePATTrackCandidates=MakePATTrackCandidates()


class MakeTrackCandidates(ConfigToolBase):
    """ Create selected tracks and a candidate hypothesis on AOD:
    """
    _label='makeTrackCandidates'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'label','TrackCands', "output collection will be <'patAOD'+label>")
        self.addParameter(self._defaultParameters,'tracks',cms.InputTag('generalTracks'), 'input tracks')
        self.addParameter(self._defaultParameters,'particleType','pi+', 'particle type (for mass)')
        self.addParameter(self._defaultParameters,'preselection','pt > 10', 'preselection cut on the AOD candidates')
        self.addParameter(self._defaultParameters,'selection','pt > 10', 'selection cut on the PAT candidates (for the selectedLayer1Candidate collection)')
        self.addParameter(self._defaultParameters,'isolation',{'tracker':0.3, 'ecalTowers':0.3, 'hcalTowers':0.3}, "isolation to use (as 'source': value of dR)\ntracker     : as muon iso from tracks\necalTowers  : as muon iso from calo tower\nhcalTowers  : as muon iso from calo towers",allowedValues=['tracker','ecalTowers','hcalTowers'])
        self.addParameter(self._defaultParameters,'isoDeposits',['tracker','ecalTowers','hcalTowers'], 'iso deposits')
        self.addParameter(self._defaultParameters,'mcAs',None, "eplicate mc match as the one used by PAT on this AOD collection (None=no mc match); choose 'photon', 'electron', 'muon', 'tau','jet', 'met' as input string",Type=str,allowedValues=['photon', 'electron', 'muon', 'tau','jet', 'met', None], acceptNoneValue = True)

        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 label        = None,
                 tracks       = None,
                 particleType = None,
                 preselection = None,
                 selection    = None,
                 isolation    = None,
                 isoDeposits  = None,
                 mcAs         = None) :
        if label  is None:
            label=self._defaultParameters['label'].value
        if tracks is None:
            tracks=self._defaultParameters['tracks'].value
        if particleType is None:
            particleType=self._defaultParameters['particleType'].value
        if preselection is None:
            preselection=self._defaultParameters['preselection'].value
        if selection is None:
            selection=self._defaultParameters['selection'].value
        if isolation is None:
            isolation=self._defaultParameters['isolation'].value
        if isoDeposits is None:
            isoDeposits=self._defaultParameters['isoDeposits'].value
        if mcAs is None:
            mcAs=self._defaultParameters['mcAs'].value
        self.setParameter('label',label)
        self.setParameter('tracks',tracks)
        self.setParameter('particleType',particleType)
        self.setParameter('preselection',preselection)
        self.setParameter('selection',selection)
        self.setParameter('isolation',isolation)
        self.setParameter('isoDeposits',isoDeposits)
        self.setParameter('mcAs',mcAs,True)
        self.apply(process)

    def toolCode(self, process):
        label=self._parameters['label'].value
        tracks=self._parameters['tracks'].value
        particleType=self._parameters['particleType'].value
        preselection=self._parameters['preselection'].value
        selection=self._parameters['selection'].value
        isolation=self._parameters['isolation'].value
        isoDeposits=self._parameters['isoDeposits'].value
        mcAs=self._parameters['mcAs'].value

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

makeTrackCandidates=MakeTrackCandidates()

from FWCore.GuiBrowsers.ConfigToolBase import *

class AddElectronUserIsolation(ConfigToolBase):

    """ add userIsolation to patElectron
    """
    _label='addElectronUserIsolation'    
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'isolationTypes',['All'],'List of predefined userIsolation types to be added; possible values are [\'Tracker\',\'Ecal\',\'Hcal\'] or just [\'All\']', allowedValues=['Tracker','Ecal','Hcal','All'])
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ''

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,isolationTypes=None) :
        if  isolationTypes is None:
            isolationTypes=self._defaultParameters['isolationTypes'].value 
        self.setParameter('isolationTypes',isolationTypes)
        self.apply(process) 
        
    def toolCode(self, process):                
        isolationTypes=self._parameters['isolationTypes'].value

        # includes to fix fastsim problems
        from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import eleIsoDepositTk, eleIsoDepositEcalFromHits, eleIsoDepositHcalFromTowers
        from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDeposits_cff import eleIsoFromDepsTk, eleIsoFromDepsEcalFromHitsByCrystal, eleIsoFromDepsHcalFromTowers
        
        eleIsoDepositEcalFromHits.ExtractorPSet.barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB")
        eleIsoDepositEcalFromHits.ExtractorPSet.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE")
        
        # key to define the parameter sets
        isolationKey=0
        # add pre-requisits to the electron
        for obj in range(len(isolationTypes)):
            if ( isolationTypes[obj] == 'Tracker' or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Electron for Tracker"
                print " -> to access this information call pat::Electron::userIsolation(pat::TrackIso) in your analysis code <-"
                isolationKey=isolationKey+1
                from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import patElectronTrackIsolation
                process.patElectronTrackIsolation
                process.patDefaultSequence.replace( process.patElectrons, process.patElectronTrackIsolation*process.patElectrons )
                
            if ( isolationTypes[obj] == 'Ecal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Electron for Ecal"
                print " -> to access this information call pat::Electron::userIsolation(pat::EcalIso ) in your analysis code <-"
                isolationKey=isolationKey+10
                from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import patElectronEcalIsolation
                process.patElectronEcalIsolation            
                process.patDefaultSequence.replace( process.patElectrons, process.patElectronEcalIsolation*process.patElectrons )
                
            if ( isolationTypes[obj] == 'Hcal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Electron for Hcal"
                print " -> to access this information call pat::Electron::userIsolation(pat::HcalIso ) in your analysis code <-"
                isolationKey=isolationKey+100
                from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import patElectronHcalIsolation            
                process.patElectronHcalIsolation = patElectronHcalIsolation
                process.patDefaultSequence.replace( process.patElectrons, process.patElectronHcalIsolation*process.patElectrons )  
                
        # do the corresponding replacements in the pat electron
        if ( isolationKey ==   1 ):
            # tracker
            process.patElectrons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("eleIsoDepositTk"),
            )
            process.patElectrons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsTk"),
                ),
            )
        if ( isolationKey ==  10 ):
            # ecal
            process.patElectrons.isoDeposits = cms.PSet(
                ecal    = cms.InputTag("eleIsoDepositEcalFromHits"),
            )
            process.patElectrons.userIsolation = cms.PSet(
                ecal = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsEcalFromHitsByCrystal"),
                ),
            )
        if ( isolationKey == 100 ):
            # hcal
            process.patElectrons.isoDeposits = cms.PSet(
                hcal    = cms.InputTag("eleIsoDepositHcalFromTowers"),
            )
            process.patElectrons.userIsolation = cms.PSet(
                hcal = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsHcalFromTowers"),
                ),
            )
        if ( isolationKey ==  11 ):
            # ecal + tracker
            process.patElectrons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("eleIsoDepositTk"),
                ecal    = cms.InputTag("eleIsoDepositEcalFromHits"),
            )
            process.patElectrons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsTk"),
                ),
                ecal = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsEcalFromHitsByCrystal"),
                ),
            )
        if ( isolationKey == 101 ):
            # hcal + tracker
            process.patElectrons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("eleIsoDepositTk"),
                hcal    = cms.InputTag("eleIsoDepositHcalFromTowers"),
            )
            process.patElectrons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsTk"),
                ),
                hcal = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsHcalFromTowers"),
                ),
            )
        if ( isolationKey == 110 ):
            # hcal + ecal
            process.patElectrons.isoDeposits = cms.PSet(
                ecal    = cms.InputTag("eleIsoDepositEcalFromHits"),
                hcal    = cms.InputTag("eleIsoDepositHcalFromTowers"),
            )
            process.patElectrons.userIsolation = cms.PSet(
                ecal = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsEcalFromHitsByCrystal"),
                ),
                hcal = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsHcalFromTowers"),
                ),
            )
        if ( isolationKey == 111 ):
            # hcal + ecal + tracker 
            process.patElectrons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("eleIsoDepositTk"),
                ecal    = cms.InputTag("eleIsoDepositEcalFromHits"),
                hcal    = cms.InputTag("eleIsoDepositHcalFromTowers"),
            )
            process.patElectrons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsTk"),
                ),
                ecal = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsEcalFromHitsByCrystal"),
                ),
                hcal = cms.PSet(
                src = cms.InputTag("eleIsoFromDepsHcalFromTowers"),
                ),
            )


addElectronUserIsolation=AddElectronUserIsolation()

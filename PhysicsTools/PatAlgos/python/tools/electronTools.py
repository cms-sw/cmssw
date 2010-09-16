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

        # load the needed modules
        process.load("PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff")
        
        # configure the electrons to read the isolations 
        for obj in range(len(isolationTypes)):
            if ( isolationTypes[obj] == 'Tracker' or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Electron for Tracker"
                print " -> to access this information call pat::Electron::userIsolation(pat::TrackIso) in your analysis code <-"
                from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import patElectronTrackIsolation
                process.patElectronTrackIsolation
                process.patDefaultSequence.replace( process.patElectrons, process.patElectronTrackIsolation*process.patElectrons )
                process.patElectrons.isoDeposits.tracker = cms.InputTag("eleIsoDepositTk")
                process.patElectrons.userIsolation.tracker = cms.PSet(src = cms.InputTag("eleIsoFromDepsTk"))
            if ( isolationTypes[obj] == 'Ecal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Electron for Ecal"
                print " -> to access this information call pat::Electron::userIsolation(pat::EcalIso ) in your analysis code <-"
                from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import patElectronEcalIsolation
                process.patElectronEcalIsolation            
                process.patDefaultSequence.replace( process.patElectrons, process.patElectronEcalIsolation*process.patElectrons )
                process.patElectrons.isoDeposits.ecal = cms.InputTag("eleIsoDepositEcalFromHits")
                process.patElectrons.userIsolation.ecal = cms.PSet(src = cms.InputTag("eleIsoFromDepsEcalFromHitsByCrystal"))
            if ( isolationTypes[obj] == 'Hcal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Electron for Hcal"
                print " -> to access this information call pat::Electron::userIsolation(pat::HcalIso ) in your analysis code <-"
                from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import patElectronHcalIsolation            
                process.patElectronHcalIsolation = patElectronHcalIsolation
                process.patDefaultSequence.replace( process.patElectrons, process.patElectronHcalIsolation*process.patElectrons )  
                process.patElectrons.isoDeposits.hcal = cms.InputTag("eleIsoDepositHcalFromTowers")
                process.patElectrons.userIsolation.hcal = cms.PSet(src = cms.InputTag("eleIsoFromDepsHcalFromTowers"))

addElectronUserIsolation=AddElectronUserIsolation()

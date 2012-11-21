from FWCore.GuiBrowsers.ConfigToolBase import *

class AddPhotonUserIsolation(ConfigToolBase):

    """ add userIsolation to patPhoton
    """
    _label='addPhotonUserIsolation'    
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

        from RecoEgamma.EgammaIsolationAlgos.gamIsoDeposits_cff import gamIsoDepositTk, gamIsoDepositEcalFromHits, gamIsoDepositHcalFromTowers
        from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff import gamIsoFromDepsTk, gamIsoFromDepsEcalFromHits, gamIsoFromDepsHcalFromTowers
        
        gamIsoDepositEcalFromHits.ExtractorPSet.barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB")
        gamIsoDepositEcalFromHits.ExtractorPSet.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE")
        
        # key to define the parameter sets
        isolationKey=0
        # add pre-requisits to the photon
        for obj in range(len(isolationTypes)):
            if ( isolationTypes[obj] == 'Tracker' or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Photon for Tracker"
                print " -> to access this information call pat::Photon::userIsolation(pat::TrackIso) in your analysis code <-"
                isolationKey=isolationKey+1
                from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import patPhotonTrackIsolation
                process.patPhotonTrackIsolation
                process.patDefaultSequence.replace( process.patPhotons, process.patPhotonTrackIsolation*process.patPhotons )
                
            if ( isolationTypes[obj] == 'Ecal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Photon for Ecal"
                print " -> to access this information call pat::Photon::userIsolation(pat::EcalIso ) in your analysis code <-"
                isolationKey=isolationKey+10
                from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import patPhotonEcalIsolation
                process.patPhotonEcalIsolation            
                process.patDefaultSequence.replace( process.patPhotons, process.patPhotonEcalIsolation*process.patPhotons )
                
            if ( isolationTypes[obj] == 'Hcal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Photon for Hcal"
                print " -> to access this information call pat::Photon::userIsolation(pat::HcalIso ) in your analysis code <-"
                isolationKey=isolationKey+100
                from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import patPhotonHcalIsolation            
                process.patPhotonHcalIsolation = patPhotonHcalIsolation
                process.patDefaultSequence.replace( process.patPhotons, process.patPhotonHcalIsolation*process.patPhotons )  
                
        # do the corresponding replacements in the pat photon
        if ( isolationKey ==   1 ):
            # tracker
            process.patPhotons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("gamIsoDepositTk"),
            )
            process.patPhotons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsTk"),
                ),
            )
        if ( isolationKey ==  10 ):
            # ecal
            process.patPhotons.isoDeposits = cms.PSet(
                ecal    = cms.InputTag("gamIsoDepositEcalFromHits"),
            )
            process.patPhotons.userIsolation = cms.PSet(
                ecal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsEcalFromHits"),
                ),
            )
        if ( isolationKey == 100 ):
            # hcal
            process.patPhotons.isoDeposits = cms.PSet(
                hcal    = cms.InputTag("gamIsoDepositHcalFromTowers"),
            )
            process.patPhotons.userIsolation = cms.PSet(
                hcal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsHcalFromTowers"),
                ),
            )
        if ( isolationKey ==  11 ):
            # ecal + tracker
            process.patPhotons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("gamIsoDepositTk"),
                ecal    = cms.InputTag("gamIsoDepositEcalFromHits"),
            )
            process.patPhotons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsTk"),
                ),
                ecal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsEcalFromHits"),
                ),
            )
        if ( isolationKey == 101 ):
            # hcal + tracker
            process.patPhotons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("gamIsoDepositTk"),
                hcal    = cms.InputTag("gamIsoDepositHcalFromTowers"),
            )
            process.patPhotons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsTk"),
                ),
                hcal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsHcalFromTowers"),
                ),
            )
        if ( isolationKey == 110 ):
            # hcal + ecal
            process.patPhotons.isoDeposits = cms.PSet(
                ecal    = cms.InputTag("gamIsoDepositEcalFromHits"),
                hcal    = cms.InputTag("gamIsoDepositHcalFromTowers"),
            )
            process.patPhotons.userIsolation = cms.PSet(
                ecal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsEcalFromHits"),
                ),
                hcal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsHcalFromTowers"),
                ),
            )
        if ( isolationKey == 111 ):
            # hcal + ecal + tracker
            process.patPhotons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("gamIsoDepositTk"),
                ecal    = cms.InputTag("gamIsoDepositEcalFromHits"),
                hcal    = cms.InputTag("gamIsoDepositHcalFromTowers"),
            )
            process.patPhotons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsTk"),
                ),
                ecal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsEcalFromHits"),
                ),
                hcal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsHcalFromTowers"),
                ),
            )


addPhotonUserIsolation=AddPhotonUserIsolation()

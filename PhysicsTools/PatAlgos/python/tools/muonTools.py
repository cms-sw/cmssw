from FWCore.GuiBrowsers.ConfigToolBase import *


class AddMuonUserIsolation(ConfigToolBase):

    """ add userIsolation to patMuon
    """
    _label='addMuonUserIsolation'    
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

        # key to define the parameter sets
        isolationKey=0
        # add pre-requisits to the muon
        for obj in range(len(isolationTypes)):
            if ( isolationTypes[obj] == 'Tracker' or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Muon for Tracker"
                print " -> to access this information call pat::Muon::userIsolation(pat::TrackIso) in your analysis code <-"
                isolationKey=isolationKey+1
                
            if ( isolationTypes[obj] == 'Ecal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Muon for Ecal"
                print " -> to access this information call pat::Muon::userIsolation(pat::EcalIso ) in your analysis code <-"
                isolationKey=isolationKey+10
                
            if ( isolationTypes[obj] == 'Hcal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Muon for Hcal"
                print " -> to access this information call pat::Muon::userIsolation(pat::HcalIso ) in your analysis code <-"
                isolationKey=isolationKey+100

        # do the corresponding replacements in the pat muon
        if ( isolationKey ==   1 ):
            # tracker
            process.patMuons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("muIsoDepositTk"),
            )
            process.patMuons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("muIsoDepositTk"),
                deltaR = cms.double(0.3)
                ),
            )
        if ( isolationKey ==  10 ):
            # ecal
            process.patMuons.isoDeposits = cms.PSet(
                ecal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
            )
            process.patMuons.userIsolation = cms.PSet(
                ecal = cms.PSet(
                src = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                deltaR = cms.double(0.3)
                ),
            )
        if ( isolationKey == 100 ):
            # hcal
            process.patMuons.isoDeposits = cms.PSet(
                hcal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
            )
            process.patMuons.userIsolation = cms.PSet(
                hcal = cms.PSet(
                src = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                deltaR = cms.double(0.3)
                ),
            )
        if ( isolationKey ==  11 ):
            # ecal + tracker
            process.patMuons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("muIsoDepositTk"),
                ecal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
            )
            process.patMuons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("muIsoDepositTk"),
                deltaR = cms.double(0.3)
                ),
                ecal = cms.PSet(
                src = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                deltaR = cms.double(0.3)
                ),
            )
        if ( isolationKey == 101 ):
            # hcal + tracker
            process.patMuons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("muIsoDepositTk"),
                hcal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
            )
            process.patMuons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("muIsoDepositTk"),
                deltaR = cms.double(0.3)
                ),
                hcal = cms.PSet(
                src = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                deltaR = cms.double(0.3)
                ),
            )
        if ( isolationKey == 110 ):
            # hcal + ecal
            process.patMuons.isoDeposits = cms.PSet(
                ecal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                hcal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
            )
            process.patMuons.userIsolation = cms.PSet(
                ecal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsEcalFromHits"),
                ),
                hcal = cms.PSet(
                src = cms.InputTag("gamIsoFromDepsHcalFromTowers"),
                ),
            )
        if ( isolationKey == 111 ):
            # hcal + ecal + tracker
            process.patMuons.isoDeposits = cms.PSet(
                tracker = cms.InputTag("muIsoDepositTk"),
                ecal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                hcal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
            )
            process.patMuons.userIsolation = cms.PSet(
                tracker = cms.PSet(
                src = cms.InputTag("muIsoDepositTk"),
                deltaR = cms.double(0.3)
                ),
                ecal = cms.PSet(
                src = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                deltaR = cms.double(0.3)
                ),
                hcal = cms.PSet(
                src = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                deltaR = cms.double(0.3)
                ),
            )


addMuonUserIsolation=AddMuonUserIsolation()

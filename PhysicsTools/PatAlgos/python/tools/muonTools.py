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

        # add pre-requisits to the muon
        for obj in range(len(isolationTypes)):
            if ( isolationTypes[obj] == 'Tracker' or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Muon for Tracker"
                print " -> to access this information call pat::Muon::userIsolation(pat::TrackIso) in your analysis code <-"
                process.patMuons.isoDeposits.tracker = cms.InputTag("muIsoDepositTk")
                process.patMuons.userIsolation.tracker = cms.PSet( src = cms.InputTag("muIsoDepositTk"), deltaR = cms.double(0.3) )
            if ( isolationTypes[obj] == 'Ecal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Muon for Ecal"
                print " -> to access this information call pat::Muon::userIsolation(pat::EcalIso ) in your analysis code <-"
                process.patMuons.isoDeposits.ecal = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal")
                process.patMuons.userIsolation.ecal = cms.PSet( src = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"), deltaR = cms.double(0.3) )
            if ( isolationTypes[obj] == 'Hcal'    or isolationTypes[obj] == 'All'):
                print "adding predefined userIsolation to pat::Muon for Hcal"
                print " -> to access this information call pat::Muon::userIsolation(pat::HcalIso ) in your analysis code <-"
                process.patMuons.isoDeposits.hcal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal")
                process.patMuons.userIsolation.hcal = cms.PSet(src = cms.InputTag( "muIsoDepositCalByAssociatorTowers","hcal"), deltaR = cms.double(0.3) )

addMuonUserIsolation=AddMuonUserIsolation()

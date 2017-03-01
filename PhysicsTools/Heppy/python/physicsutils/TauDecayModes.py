
class TauDecayModes( object ): 

    def __init__(self):
        self._decayModes() 
        
    def _decayModes(self):
        '''Builds the internal dictionaries from the enum defined in 
        http://cmslxr.fnal.gov/lxr/source/DataFormats/TauReco/interface/PFTau.h'''
        tmp = [
            'kNull',
            'kOneProng0PiZero',
            'kOneProng1PiZero',
            'kOneProng2PiZero',
            'kOneProng3PiZero',
            'kOneProngNPiZero',
            'kTwoProng0PiZero',
            'kTwoProng1PiZero',
            'kTwoProng2PiZero',
            'kTwoProng3PiZero',
            'kTwoProngNPiZero',
            'kThreeProng0PiZero',
            'kThreeProng1PiZero',
            'kThreeProng2PiZero',
            'kThreeProng3PiZero',
            'kThreeProngNPiZero',
            'kRareDecayMode'
            ]
        self.decayModes = dict( (index-1, name) for index, name in enumerate( tmp ) )
        self.decayModeNames = dict( (value, key) for key, value \
                                    in self.decayModes.iteritems() )

    def intToName( self, anInt ):
        '''Returns the decay mode name corresponding to an int.'''
        return self.decayModes[ anInt ]

    def nameToInt( self, aName ):
        '''Returns the decay mode int corresponding to a name.'''
        return self.decayModeNames[ aName ]      
    
    def __str__(self):
        return str( self.decayModes )

    def genDecayModeInt(self, daughters):
        dm = self.genDecayMode(daughters)
        return self.translateGenModeToInt(dm)

    def genDecayModeFromJetInt(self, c):
        dm = self.genDecayModeFromGenJet(c)
        return self.translateGenModeToInt(dm)

    def translateGenModeToInt(self, dm):
        if dm in self.decayModeNames:
            return self.nameToInt(dm)
        elif dm == 'electron':
            return -11
        elif dm == 'muon':
            return -13
        elif dm == 'kOneProngOther':
            return 100 # one-prong + 100
        elif dm == 'kThreeProngOther':
            return 110 # three-prong + 100
        return -99

    @staticmethod
    def genDecayModeFromGenJet(c):
        ''' Returns generated tau decay mode. Needs to be called on genJet
        as stored in pat::Tau, if available.

        Translated from PhysicsTools/JetMCUtils/interface/JetMCTag.h,
        which is not available in FWlite.
        '''

        daughters = c.daughterPtrVector()

        return TauDecayModes.genDecayMode(daughters)

    @staticmethod
    def genDecayMode(daughters):
        ''' Returns the generated tau decay mode based on a passed list of all
        final daughters before further decay (as contained in miniAOD).
        '''
        numElectrons = 0
        numMuons = 0
        numChargedHadrons = 0
        numNeutralHadrons = 0
        numPhotons = 0
  
        for daughter in daughters:
            pdg_id = abs(daughter.pdgId())
            if pdg_id == 22:
                numPhotons += 1
            elif pdg_id == 11:
                numElectrons +=1
            elif pdg_id == 13:
                numMuons += 1
            else:
                if daughter.charge() != 0:
                    numChargedHadrons += 1
                elif pdg_id not in [12, 14, 16]:
                    numNeutralHadrons += 1

        if numElectrons == 1:
            return "electron"
        if numMuons == 1:
            return "muon"

        if numChargedHadrons == 1:
            if numNeutralHadrons != 0:
                return "kOneProngOther"
            if numPhotons == 0:
                return "kOneProng0PiZero"
            elif numPhotons == 2:
                return "kOneProng1PiZero"
            elif numPhotons == 4:
                return "kOneProng2PiZero"
            elif numPhotons == 6:
                return "kOneProng3PiZero"
            else:
                return "kOneProngNPiZero"
        elif numChargedHadrons == 3:
            if numNeutralHadrons != 0:
                return "kThreeProngOther"
            if numPhotons == 0:
                return "kThreeProng0PiZero"
            elif numPhotons == 2:
                return "kThreeProng1PiZero"
            elif numPhotons == 4:
                return "kThreeProng2PiZero"
            elif numPhotons == 6:
                return "kThreeProng3PiZero"
            else:
                return "kThreeProngNPiZero"

        return "kRareDecayMode"

tauDecayModes = TauDecayModes()

if __name__ == '__main__':

    dec = TauDecayModes()
    print dec

    print 0, dec.intToName(0)
    print 'kThreeProng0PiZero', dec.nameToInt('kThreeProng0PiZero')

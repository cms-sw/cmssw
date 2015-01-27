print 'importing'

class TauDecayModes( object ): 

    def __init__(self):
        self._decayModes() 
        
    def _decayModes(self):
        '''Builds the internal dictionaries from the enum defined in 
        http://cmslxr.fnal.gov/lxr/source/DataFormats/TauReco/interface/PFTau.h'''
        tmp = [
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
        self.decayModes = dict( (index, name) for index,name in enumerate( tmp ) )
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

tauDecayModes = TauDecayModes()

if __name__ == '__main__':

    dec = TauDecayModes()
    print dec

    print 0, dec.intToName(0)
    print 'kThreeProng0PiZero', dec.nameToInt('kThreeProng0PiZero')

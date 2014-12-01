# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

def printWeights( weights ):
    for key, value in weights.iteritems():
        print key
        print value 

class Weight( object ):
    '''make names uniform wrt Component.

    COLIN: messy... should I have several types of weight (base, data, mc)?
    COLIN: need to add other weighting factors'''

    FBINV = 1000.
     
    def __init__(self, genNEvents, xSection, genEff,
                 intLumi = FBINV, addWeight=1):
        self.genNEvents = int(genNEvents)
        if xSection is not None:
            self.xSection = float(xSection)
        else:
            self.xSection = None
        self.genEff = float(genEff)
        if intLumi is not None:
            self.intLumi = float(intLumi)
        else:
            self.intLumi = Weight.FBINV
        self.addWeight = float(addWeight)
        
    def GetWeight(self):
        '''Return the weight'''
        if self.xSection is None:
            # data
            return 1 
        else:
            # MC
            return self.addWeight * self.xSection * self.intLumi / ( self.genNEvents * self.genEff) 

    def SetIntLumi(self, lumi):
        '''Set integrated luminosity.'''
        self.dict['intLumi'] = lumi

    def __str__(self):
        if self.xSection is None:          
            return ' intLumi = %5.2f, addWeight = %3.2f' \
               % ( self.intLumi,
                   self.addWeight )
        else:
            return ' genN = %d, xsec = %5.5f pb, genEff = %2.2f, intLumi = %5.2f, addWeight = %3.2f -> weight = %3.5f' \
                   % ( self.genNEvents,
                       self.xSection,
                       self.genEff,
                       self.intLumi,
                       self.addWeight,
                       self.GetWeight() )
        

from CMGTools.VVResonances.analyzers.EventInterpretationBase import *
from CMGTools.VVResonances.tools.Pair import Pair

class LNuJJ( EventInterpretationBase ):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(LNuJJ,self).__init__(cfg_ana, cfg_comp, looperName)

    def process(self, event):
        super(LNuJJ,self).process(event)

        output=[]
        #read the W
        for w in event.LNu:
            leptons = [w.leg1]
            cleanedPackedCandidates=self.removeLeptonFootPrint(leptons,event.packedCandidatesForJets)
            if self.cfg_ana.doCHS:
                cleanedPackedCandidates = filter(lambda x: x.fromPV(0) ,cleanedPackedCandidates)


            #apply selections
            selectedFatJets = self.makeFatJets(cleanedPackedCandidates)
            if self.isMC:
                self.matchSubJets(selectedFatJets,event.genwzquarks)

            for fat in selectedFatJets:
                VV = Pair(w,fat)
                if self.selectPair(VV):
                    selected = {'pair':VV}
                    remainingCands =self.removeJetFootPrint([fat],cleanedPackedCandidates)
                    selected['satelliteJets']=self.makeSatelliteJets(remainingCands)
                    #add VBF info
                    self.vbfTopology(selected)
                    output.append(selected)                   
#                    import pdb;pdb.set_trace()
        if len(output)>0:
            output = sorted(output,key = lambda x: x['pair'].mass(),reverse=True)
        setattr(event,'LNuJJ'+self.cfg_ana.suffix,output)
        return True

from PhysicsTools.HeppyCore.utils.deltar import deltaR


class OverlapCleaner(object):
    def __init__(self,collection,dr,sourcePdgId,targetPdgId,targetID):
        self.collection=collection
        self.dr = dr
        self.sourcePdgId=sourcePdgId
        self.targetPdgId=targetPdgId
        self.targetID = targetID

    def __call__(self,object):
        hasOverlap=False
        if abs(object.pdgId()) ==self.sourcePdgId:
            for p in self.collection:
                if abs(p.pdgId()) ==self.targetPdgId:
                    if self.targetID(p)==True:
                        if deltaR(object.eta(),object.phi(),p.eta(),p.phi())<self.dr:
                            hasOverlap=True

        if hasOverlap:
            return False                 
        else:
            return True

                          
                

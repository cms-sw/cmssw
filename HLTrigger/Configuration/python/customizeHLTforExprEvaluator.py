import FWCore.ParameterSet.Config as cms

class RunCutSyntaxFixer(object):
    def __init__(self):
        self.visited = {}

    def __call__(self,name,module):
        if name in self.visited: return
        if hasattr(module,'vertexCut') and type(module.vertexCut) is cms.string:
            self.visited[name] = True            
            temp = module.vertexCut.value()
            if temp.find('obj.') != -1: return # already converted
            temp = temp.replace('tracksSize','obj.tracksSize()')
            module.vertexCut = cms.string(temp)
        if hasattr(module,'cut') and type(module.cut) is cms.string:
            self.visited[name] = True
            temp = module.cut.value()
            if temp.find('obj.') != -1: return # already converted
            temp = temp.replace(' & ',' && ')
            temp = temp.replace('pt','obj.pt()')
            temp = temp.replace('abs','std::abs')
            temp = temp.replace('(eta)','(obj.eta())')
            temp = temp.replace('n90','obj.n90()')
            temp = temp.replace('n60','obj.n60()')
            temp = temp.replace('towersArea','obj.towersArea()')
            temp = temp.replace('emEnergyFraction','obj.emEnergyFraction()')
            temp = temp.replace('energyFractionHadronic','obj.energyFractionHadronic()')
            temp = temp.replace('isFake','obj.isFake()')
            temp = temp.replace('ndof','obj.ndof()')
            temp = temp.replace('(z)','(obj.z())')
            temp = temp.replace('position.Rho','obj.position().Rho()')
            module.cut = cms.string(temp)
        if hasattr(module,'ranking') and type(module.ranking) is cms.VPSet:
            for submod in module.ranking:
                if hasattr(submod,'selection') and type(submod.selection) is cms.string:
                    self.visited[name] = True
                    temp = submod.selection.value()
                    if temp.find('obj.') != -1: return # already converted
                    temp = temp.replace("algoIs(\'kChargedPFCandidate\')",
                                        'obj.algoIs(reco::PFRecoTauChargedHadron::kChargedPFCandidate)')
                    temp = temp.replace("algoIs(\'kStrips\')",
                                        'obj.algoIs(reco::RecoTauPiZero::kStrips)')
                    submod.selection = cms.string(temp)
        if hasattr(module,'outputSelection') and type(module.outputSelection) is cms.string:
            self.visited[name] = True
            temp = module.outputSelection.value()
            if temp.find('obj.') != -1: return # already converted
            temp = temp.replace('pt','obj.pt()')                     
            module.outputSelection = cms.string(temp)

            

runCutSyntaxFixer = RunCutSyntaxFixer()

def customizeHLTforExprEvaluator(process):
    for name in process.producers:
        runCutSyntaxFixer(name,process.producers[name])
    for name in process.filters:
        runCutSyntaxFixer(name,process.filters[name])
    for name in process.analyzers:
        runCutSyntaxFixer(name,process.analyzers[name])
    return process

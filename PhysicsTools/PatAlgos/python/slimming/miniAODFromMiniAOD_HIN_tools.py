import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers  import getPatAlgosToolsTask, addToProcessAndTask

def miniAODFromMiniAOD_customizeCommon(process):

    task = getPatAlgosToolsTask(process)
    cols = []

    ###########################################################################
    # Update egamma regression
    ###########################################################################
    def _updateEGRegression(process):
        from RecoEgamma.EgammaTools.regressionModifier_cfi import regressionModifier
        for obj in ['Electron','Photon']:
            setattr(process,f'slimmed{obj}s',cms.EDProducer(f"Modified{obj}Producer",
                src = cms.InputTag(f'slimmed{obj}s::@skipCurrentProcess'),
                modifierConfig = cms.PSet(modifications = cms.VPSet(regressionModifier))
            ))
            task.add(getattr(process,f'slimmed{obj}s'))
            cols.append(f'slimmed{obj}s')

    from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
    pp_on_PbPb_run3.toModify(process, _updateEGRegression)

    ###########################################################################
    # Add secondary vertex
    ###########################################################################
    def _addSecondaryVertex(process,n=''):
        process.load('TrackingTools.TransientTrack.TransientTrackBuilder_cfi')
        import RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff as _sv
        if n == 'Negative':
            import RecoVertex.AdaptiveVertexFinder.inclusiveNegativeVertexing_cff as _sv
        for mod in [f'inclusiveCandidate{n}VertexFinder',f'candidate{n}VertexArbitrator']:
            setattr(process,mod,getattr(_sv,mod).clone(
                tracks = "packedPFCandidates",
                primaryVertices = "offlineSlimmedPrimaryVertices"
            ))
        getattr(process,f'inclusiveCandidate{n}VertexFinder').minHits = 10
        getattr(process,f'inclusiveCandidate{n}VertexFinder').minPt = 1.0
        setattr(process,f'candidate{n}VertexMerger',getattr(_sv,f'candidate{n}VertexMerger').clone())
        setattr(process,f'slimmed{n}SecondaryVertices',getattr(_sv,f'inclusiveCandidate{n}SecondaryVertices').clone())
        for mod in [f'inclusiveCandidate{n}VertexFinder',f'candidate{n}VertexMerger',
                    f'candidate{n}VertexArbitrator',f'slimmed{n}SecondaryVertices']:
            task.add(getattr(process,mod))
        cols.append(f'slimmed{n}SecondaryVertices')

    def _addNegativeSecondaryVertex(process):
        return _addSecondaryVertex(process,n='Negative')

    from Configuration.Eras.Modifier_pp_on_PbPb_run3_2023_cff import pp_on_PbPb_run3_2023
    from Configuration.Eras.Modifier_pp_on_PbPb_run3_2024_cff import pp_on_PbPb_run3_2024
    (pp_on_PbPb_run3_2023 | pp_on_PbPb_run3_2024).toModify(process, _addSecondaryVertex)
    pp_on_PbPb_run3.toModify(process, _addNegativeSecondaryVertex)

    ###########################################################################
    # Add centrality
    ###########################################################################
    def _addCentrality(process):
        process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
        process.centralityBin.Centrality = "hiCentrality"
        process.centralityBin.centralityVariable = "HFtowers"
        task.add(process.centralityBin)
        cols.append('centralityBin')

    pp_on_PbPb_run3.toModify(process, _addCentrality)

    ###########################################################################
    # Define output collections
    ###########################################################################
    outputCommands = []
    for new_collection_to_keep in cols:
        new_collection_to_keep += '_*' if not '_' in new_collection_to_keep else ''
        outputCommands += [
            f'drop *_{new_collection_to_keep}_*',
            f'keep *_{new_collection_to_keep}_{process.name_()}']

    mini_output = None
    for out_name in process.outputModules_().keys():
        if out_name.startswith('MINIAOD'):
            mini_output = getattr(process, out_name)
            break
    if mini_output:
        mini_output.outputCommands += outputCommands

    if hasattr(process,'SKIMStreamPbPbEW'):
        process.SKIMStreamPbPbEW.outputCommands += outputCommands

    return process

def miniAODFromMiniAOD_customizeData(process):
    return process

def miniAODFromMiniAOD_customizeMC(process):
    return process

def miniAODFromMiniAOD_customizeAllData(process):
    process = miniAODFromMiniAOD_customizeData(process)
    process = miniAODFromMiniAOD_customizeCommon(process)
    return process

def miniAODFromMiniAOD_customizeAllMC(process):
    process = miniAODFromMiniAOD_customizeMC(process)
    process = miniAODFromMiniAOD_customizeCommon(process)
    return process

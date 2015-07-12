from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import * 
from CMGTools.VVResonances.analyzers.vvTypes  import * 
from CMGTools.VVResonances.analyzers.Skimmer  import * 
import PhysicsTools.HeppyCore.framework.config as cfg

vvSkimmer = cfg.Analyzer(
    Skimmer,
    name='vvSkimmer',
    required = ['LNuJJ']
)


vvTreeProducer = cfg.Analyzer(
     AutoFillTreeProducer, name='vvTreeProducer',
     vectorTree = True,
     saveTLorentzVectors = False,  # can set to True to get also the TLorentzVectors, but trees will be bigger
     defaultFloatType = 'F', # use Float_t for floating point
#     PDFWeights = PDFWeights,
     globalVariables = [
        NTupleVariable("nAllLNu",lambda ev: len(ev.allLNu) , int),       
        NTupleVariable("nAllLL",lambda ev: len(ev.allLL) , int),       
        NTupleVariable("nLL",lambda ev: len(ev.LL) , int),       
        NTupleVariable("nLNu",lambda ev: len(ev.LNu) , int),       
     ],
#     globalObjects = susyMultilepton_globalObjects,
     collections = {
#            "genleps"          : NTupleCollection("gen",     genParticleWithLinksType, 10, help="Generated leptons (e/mu) from W/Z decays"),                                                                                                
#            "inclusiveLeptons" : NTupleCollection("l",    leptonTypeExtra, 10, help="Inclusive Leptons"),                                                                                                
            "LNuJJ" : NTupleCollection("lnujj",LNuJJType ,5, help="VV candidate with a lepton neutrino and a fat jet")                                                                                                
#            "LNuJJCA" : NTupleCollection("lnujjca",LNuJJType ,5, help="VV candidate with a lepton neutrino and a fat jet"),                                                                                                
#            "genVBosons" : NTupleCollection("genV",     genParticleWithLinksType, 10, help="Generated V bosons"),                                                                                                
     }
)



leptonTreeProducer = cfg.Analyzer(
     AutoFillTreeProducer, name='leptonTreeProducer',
     vectorTree = True,
     saveTLorentzVectors = False,  # can set to True to get also the TLorentzVectors, but trees will be bigger
     defaultFloatType = 'F', # use Float_t for floating point
#     PDFWeights = PDFWeights,
#     globalVariables = susyMultilepton_globalVariables,
#     globalObjects = susyMultilepton_globalObjects,
     collections = {
            "genleps"          : NTupleCollection("gen",     genParticleWithLinksType, 10, help="Generated leptons (e/mu) from W/Z decays"),                                                                                                
            "inclusiveLeptons" : NTupleCollection("l",    leptonTypeExtra, 10, help="Inclusive Leptons"),                                                                                                
     }
)

#! /usr/bin/env python
# This example shows how to have multiple Tree either in the same file on in different file.
# In particular here we create a second tree producer containing only information and then, 
# cloning it in two copies, we store it both in the same file as the main tree and in separate file

import ROOT
import PhysicsTools.HeppyCore.framework.config as cfg
# avoid creating subdirs, in case subdirs are wanted the treeProducer should have different names (set name="blabla" in the config)
cfg.Analyzer.nosubdir=True

# The content of the output tree is defined here
# the definitions of the NtupleObjects are located under PhysicsTools/Heppy/pythonanalyzers/objects/autophobj.py
 
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import * 
treeProducer= cfg.Analyzer(
	class_object=AutoFillTreeProducer, 
	verbose=False, 
	vectorTree = True,
        #here the list of simple event variables (floats, int) can be specified
        globalVariables = [
             NTupleVariable("rho",  lambda ev: ev.rho, float, help="jets rho"),
        ],
        #here one can specify compound objects 
        globalObjects = {
          "met"    : NTupleObject("met",     metType, help="PF E_{T}^{miss}, after default type 1 corrections"),
        },
	collections = {
		#The following would just store the electrons and muons from miniaod without any selection or cleaning
                # only the basice particle information is saved
		#"slimmedMuons" : ( AutoHandle( ("slimmedMuons",), "std::vector<pat::Muon>" ),
                #           NTupleCollection("mu", particleType, 4, help="patMuons, directly from MINIAOD") ),
                #"slimmedElectron" : ( AutoHandle( ("slimmedElectrons",), "std::vector<pat::Electron>" ),
                #           NTupleCollection("ele", particleType, 4, help="patElectron, directly from MINIAOD") ),

		#standard dumping of objects
   	        "selectedLeptons" : NTupleCollection("leptons", leptonType, 8, help="Leptons after the preselection"),
                "selectedTaus"    : NTupleCollection("TauGood", tauType, 3, help="Taus after the preselection"),
	        "cleanJets"       : NTupleCollection("Jet",     jetType, 8, help="Cental jets after full selection and cleaning, sorted by b-tag"),
		#dump of gen objects
                "gentopquarks"    : NTupleCollection("GenTop",     genParticleType, 2, help="Generated top quarks from hard scattering"),
                "genbquarks"      : NTupleCollection("GenBQuark",  genParticleType, 2, help="Generated bottom quarks from top quark decays"),
                "genwzquarks"     : NTupleCollection("GenQuark",   genParticleType, 6, help="Generated quarks from W/Z decays"),
                "genleps"         : NTupleCollection("GenLep",     genParticleType, 6, help="Generated leptons from W/Z decays"),
                "gentauleps"      : NTupleCollection("GenLepFromTau", genParticleType, 6, help="Generated leptons from decays of taus from W/Z/h decays"),

	}
	)

#make a light weight dump containing only generator information
treeProducer2= cfg.Analyzer(
	treename="genonly",
	ignoreAnalyzerBookings=True, #we do not want trigger bits here or any other central booking
        class_object=AutoFillTreeProducer,
        verbose=False,
        vectorTree = True,
        collections = {
                #dump of gen objects
                "gentopquarks"    : NTupleCollection("GenTop",     genParticleType, 2, help="Generated top quarks from hard scattering"),
                "genbquarks"      : NTupleCollection("GenBQuark",  genParticleType, 2, help="Generated bottom quarks from top quark decays"),
                "genwzquarks"     : NTupleCollection("GenQuark",   genParticleType, 6, help="Generated quarks from W/Z decays"),
                "genleps"         : NTupleCollection("GenLep",     genParticleType, 6, help="Generated leptons from W/Z decays"),
                "gentauleps"      : NTupleCollection("GenLepFromTau", genParticleType, 6, help="Generated leptons from decays of taus from W/Z/h decays"),

        }
        )

#create a copy of tree producer with the difference that it stores it in a separate file
from copy import deepcopy 
treeProducer3 = deepcopy(treeProducer2)
treeProducer3.filter = lambda ev : len(getattr(ev,"genbquarks",[])) > 0 # select only events with b-quarks
treeProducer3.outservicename="genonlyfile"



# Import standard analyzers and take their default config
from PhysicsTools.Heppy.analyzers.objects.LeptonAnalyzer import LeptonAnalyzer
LepAna = LeptonAnalyzer.defaultConfig
from PhysicsTools.Heppy.analyzers.objects.VertexAnalyzer import VertexAnalyzer
VertexAna = VertexAnalyzer.defaultConfig
from PhysicsTools.Heppy.analyzers.objects.PhotonAnalyzer import PhotonAnalyzer
PhoAna = PhotonAnalyzer.defaultConfig
from PhysicsTools.Heppy.analyzers.objects.TauAnalyzer import TauAnalyzer
TauAna = TauAnalyzer.defaultConfig
from PhysicsTools.Heppy.analyzers.objects.JetAnalyzer import JetAnalyzer
JetAna = JetAnalyzer.defaultConfig
from PhysicsTools.Heppy.analyzers.gen.LHEAnalyzer import LHEAnalyzer 
LHEAna = LHEAnalyzer.defaultConfig
from PhysicsTools.Heppy.analyzers.gen.GeneratorAnalyzer import GeneratorAnalyzer 
GenAna = GeneratorAnalyzer.defaultConfig
from PhysicsTools.Heppy.analyzers.objects.METAnalyzer import METAnalyzer
METAna = METAnalyzer.defaultConfig
from PhysicsTools.Heppy.analyzers.core.PileUpAnalyzer import PileUpAnalyzer
PUAna = PileUpAnalyzer.defaultConfig
from PhysicsTools.Heppy.analyzers.core.TriggerBitAnalyzer import TriggerBitAnalyzer
FlagsAna = TriggerBitAnalyzer.defaultEventFlagsConfig

# Configure trigger bit analyzer
from PhysicsTools.Heppy.analyzers.core.TriggerBitAnalyzer import TriggerBitAnalyzer
TrigAna= cfg.Analyzer(
    verbose=False,
    class_object=TriggerBitAnalyzer,
    #grouping several paths into a single flag
    # v* can be used to ignore the version of a path
    triggerBits={
    'ELE':["HLT_Ele23_Ele12_CaloId_TrackId_Iso_v*","HLT_Ele32_eta2p1_WP85_Gsf_v*","HLT_Ele32_eta2p1_WP85_Gsf_v*"],
    'MU': ["HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*","HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*","HLT_IsoTkMu24_eta2p1_IterTrk02_v*","HLT_IsoTkMu24_IterTrk02_v*"],
    },
#   processName='HLT',
#   outprefix='HLT'
    #setting 'unrollbits' to true will not only store the OR for each set of trigger bits but also the individual bits
    #caveat: this does not unroll the version numbers
    unrollbits=True 
    )



#replace some parameters
LepAna.loose_muon_pt = 10

sequence = [LHEAna,FlagsAna, GenAna, PUAna,TrigAna,VertexAna,LepAna,TauAna,PhoAna,JetAna,METAna,treeProducer,treeProducer2,treeProducer3]

#use tfile service to provide a single TFile to all modules where they
#can write any root object. If the name is 'outputfile' or the one specified in treeProducer
#also the treeProducer uses this file
from PhysicsTools.HeppyCore.framework.services.tfile import TFileService 
output_service = cfg.Service(
      TFileService,
      'outputfile',
      name="outputfile",
      fname='tree.root',
      option='recreate'
    )
output_service2= cfg.Service(
      TFileService,
      'genonlyfile',
      name="genonlyfile",
      fname='treegen.root',
      option='recreate'
    )


# the following two lines are just for automatic testing
# they are not needed for running on your own samples
from PhysicsTools.Heppy.utils.miniAodFiles import miniAodFiles
testfiles=miniAodFiles()
print "Running on test file %s" % testfiles

sample = cfg.MCComponent(
#specify the file you want to run on
#    files = ["/scratch/arizzi/Hbb/CMSSW_7_2_2_patch2/src/VHbbAnalysis/Heppy/test/ZLL-8A345C56-6665-E411-9C25-1CC1DE04DF20.root"],
    files = testfiles,
    name="SingleSample", isMC=True,isEmbed=False
    )

# the following is declared in case this cfg is used in input to the heppy.py script
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
selectedComponents = [sample]
config = cfg.Config( components = selectedComponents,
                     sequence = sequence,
                     services = [output_service,output_service2],  
                     events_class = Events)

# and the following runs the process directly if running as with python filename.py  
if __name__ == '__main__':
    from PhysicsTools.HeppyCore.framework.looper import Looper 
    looper = Looper( 'Loop', config, nPrint = 5,nEvents=300) 
    looper.loop()
    looper.write()

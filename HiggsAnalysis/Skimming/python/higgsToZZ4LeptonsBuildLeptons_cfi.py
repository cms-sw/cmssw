import FWCore.ParameterSet.Config as cms

#MUON_BASE_CUT=("(isGlobalMuon &&" +
#	      " pt > 5 &&" +
#              " eta <= 2.5)");

MUON_BASE_CUT=("(isGlobalMuon)");

goodMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string(MUON_BASE_CUT)
)

#ELECTRON_BASE_CUT=("(pt > 5 &&" +
#		   " eta <= 2.5)");

ELECTRON_BASE_CUT=("");

goodElectrons = cms.EDFilter("GsfElectronRefSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string(ELECTRON_BASE_CUT),
)

# merging electrons and muons
goodLeptons = cms.EDProducer("CandViewMerger",
       src = cms.VInputTag( "goodElectrons", "goodMuons")
) 


# dileptons any flavour and charge
diLeptonsOS  = cms.EDProducer("CandViewShallowCloneCombiner",
    decay           = cms.string("goodLeptons@+ goodLeptons@-"), 
    cut             = cms.string("(mass > 0)")
)

#
diLeptonsSSplus  = cms.EDProducer("CandViewShallowCloneCombiner",
    decay           = cms.string("goodLeptons@+ goodLeptons@+"), 
    cut             = cms.string("(mass > 0 && (daughter(0).charge>0 && daughter(1).charge>0))")
)

#
diLeptonsSSminus  = cms.EDProducer("CandViewShallowCloneCombiner",
    decay           = cms.string("goodLeptons@- goodLeptons@-"), 
    cut             = cms.string("(mass > 0 && (daughter(0).charge<0 && daughter(1).charge<0))")
)
#
diLeptonsSS       = cms.EDProducer("CandViewMerger",
    src = cms.VInputTag( "diLeptonsSSplus", "diLeptonsSSminus")
) 

## Dimuons and electorn OS
diMuonsZ = cms.EDProducer("CandViewShallowCloneCombiner",
    decay           = cms.string("goodMuons@+ goodMuons@-"), 
    cut             = cms.string("(mass > 0)")
)

diElectronsZ = cms.EDProducer("CandViewShallowCloneCombiner",
    decay       = cms.string("goodElectrons@+ goodElectrons@-"), 
    cut         = cms.string("(mass > 0)")
)


higgsToZZ4LeptonsBuildLeptons = cms.Sequence(
    goodMuons        +
    goodElectrons    +
    goodLeptons      +
    diLeptonsOS      +
    diLeptonsSSplus  +
    diLeptonsSSminus +
    diLeptonsSS      +
    diMuonsZ         +
    diElectronsZ
)



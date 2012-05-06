import FWCore.ParameterSet.Config as cms

####### MUON SELECTION
MUON_CUT       = ("abs(eta)<2.5 && (isGlobalMuon || isTrackerMuon)")
DIMUON_MASSCUT = ("mass > 30")
DIMUON_KINCUT  = ("(max(daughter(0).pt(),daughter(1).pt())>20 && min(daughter(0).pt,daughter(1).pt())>10)")
goodHzzMuons = cms.EDFilter("MuonRefSelector", src = cms.InputTag("muons"), cut = cms.string(MUON_CUT) )
hzzKinDiMuons = cms.EDProducer("CandViewShallowCloneCombiner",
                               decay       = cms.string("goodHzzMuons goodHzzMuons"),
                               checkCharge = cms.bool(False),
                               cut         = cms.string(DIMUON_KINCUT)
                               )
hzzKinDiMuonsFilter = cms.EDFilter("CandViewCountFilter", src = cms.InputTag("hzzKinDiMuons"), minNumber = cms.uint32(1) )
hzzMassDiMuons = hzzKinDiMuons.clone( cut=cms.string(DIMUON_MASSCUT) )
hzzMassDiMuonsFilter = hzzKinDiMuonsFilter.clone( src = cms.InputTag("hzzMassDiMuons"))


####### ELECTRON SELECTION
ELECTRON_CUT       = ("abs(eta)<2.5")
DIELECTRON_MASSCUT = ("mass > 40")
DIELECTRON_KINCUT  = ("(max(daughter(0).pt(),daughter(1).pt())>20 && min(daughter(0).pt,daughter(1).pt())>10)")
goodHzzElectrons = cms.EDFilter("GsfElectronRefSelector", src = cms.InputTag("gsfElectrons"), cut = cms.string(ELECTRON_CUT) )
hzzKinDiElectrons = cms.EDProducer("CandViewShallowCloneCombiner",
                                   decay       = cms.string("goodHzzElectrons goodHzzElectrons"),
                                   checkCharge = cms.bool(False),
                                   cut         = cms.string(DIELECTRON_KINCUT)
                                   )
hzzKinDiElectronsFilter = cms.EDFilter("CandViewCountFilter", src = cms.InputTag("hzzKinDiElectrons"), minNumber = cms.uint32(1) )
hzzMassDiElectrons = hzzKinDiElectrons.clone( cut=cms.string(DIELECTRON_MASSCUT) )
hzzMassDiElectronsFilter = hzzKinDiElectronsFilter.clone( src = cms.InputTag("hzzMassDiElectrons") )

########## CROSS SELECTION
DILEPTON_MASSCUT = ("mass > 40")
DILEPTON_KINCUT  = ("(max(daughter(0).pt(),daughter(1).pt())>20 && min(daughter(0).pt,daughter(1).pt())>10)")
hzzKinCrossLeptons = cms.EDProducer("CandViewShallowCloneCombiner",
                                    decay       = cms.string("goodHzzElectrons goodHzzMuons"),
                                    checkCharge = cms.bool(False),
                                    cut         = cms.string(DILEPTON_KINCUT)
                                    )

hzzKinCrossLeptonsFilter = cms.EDFilter("CandViewCountFilter", src = cms.InputTag("hzzKinCrossLeptons"), minNumber = cms.uint32(1) )
hzzMassCrossLeptons = hzzKinCrossLeptons.clone( cut=cms.string(DILEPTON_MASSCUT) )
hzzMassCrossLeptonsFilter = hzzKinCrossLeptonsFilter.clone( src = cms.InputTag("hzzMassCrossLeptons") )

#     2e   2m   em
# 2e  4e   2e2m 2eem
# 2m  2m2e 4m   2mem
# em  em2e em2m emem

zz4eSequence   = cms.Sequence( goodHzzElectrons * hzzKinDiElectrons * hzzKinDiElectronsFilter * hzzMassDiElectrons * hzzMassDiElectronsFilter)
HZZ4ePath      = cms.Path( zz4eSequence )
zz2e2mSequence = cms.Sequence( goodHzzMuons * goodHzzElectrons * hzzKinDiElectrons * hzzKinDiElectronsFilter * hzzMassDiMuons * hzzMassDiMuonsFilter)
HZZ2e2mPath    = cms.Path( zz2e2mSequence ) 
zz2eemSequence = cms.Sequence( goodHzzMuons * goodHzzElectrons * hzzKinDiElectrons * hzzKinDiElectronsFilter * hzzMassCrossLeptons * hzzMassCrossLeptonsFilter )
HZZ2eemPath    = cms.Path( zz2eemSequence )

zz2m2eSequence = cms.Sequence( goodHzzMuons * goodHzzElectrons * hzzKinDiMuons * hzzKinDiMuonsFilter * hzzKinDiElectrons * hzzKinDiElectronsFilter )
HZZ2m2ePath    = cms.Path( zz2m2eSequence ) 
zz4mSequence   = cms.Sequence( goodHzzMuons * hzzKinDiMuons * hzzKinDiMuonsFilter * hzzMassDiMuons * hzzMassDiMuonsFilter)
HZZ4mPath      = cms.Path( zz4mSequence )
zz2memSequence = cms.Sequence( goodHzzMuons * goodHzzElectrons * hzzKinDiMuons * hzzKinDiMuonsFilter * hzzMassCrossLeptons * hzzMassCrossLeptonsFilter )
HZZ2memPath    = cms.Path ( zz2memSequence )

zzem2eSequence = cms.Sequence( goodHzzMuons * goodHzzElectrons * hzzKinCrossLeptons * hzzKinCrossLeptonsFilter * hzzMassDiElectrons * hzzMassDiElectronsFilter)
HZZem2ePath    = cms.Path( zzem2eSequence )
zzem2mSequence = cms.Sequence( goodHzzMuons * goodHzzElectrons * hzzKinCrossLeptons * hzzKinCrossLeptonsFilter * hzzMassDiMuons * hzzMassDiMuonsFilter )
HZZem2mPath    = cms.Path( zzem2mSequence )
zzememSequence = cms.Sequence( goodHzzMuons * goodHzzElectrons * hzzKinCrossLeptons * hzzKinCrossLeptonsFilter * hzzMassCrossLeptons * hzzMassCrossLeptonsFilter )
HZZememPath    = cms.Path( zzememSequence )

# list of paths to run
HZZPaths=(HZZ4ePath, HZZ2e2mPath, HZZ2m2ePath, HZZ4mPath, HZZem2ePath, HZZem2mPath)




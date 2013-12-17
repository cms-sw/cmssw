import FWCore.ParameterSet.Config as cms

#==================================================== FILTER EVENTS WITH WZSkims
#from DPGAnalysis.Skims.WElectronSkim_cff import * # already imported by ZElectronSkim_cff
from DPGAnalysis.Skims.ZElectronSkim_cff import *
# tagGsfSeq for Zee
#
WSkimSeq = cms.Sequence( WEnuHltFilter * ele_sequence * elecMetFilter )

#from calibration.SANDBOX.ZElectronSkimSandbox_cff import *

#===================================================== removing events with trackerDrivenOnly electrons
# if you want to filter events with trackerDrivenOnly electrons
# you should produce a collection containing the Ref to the
# trackerDrivenOnly electrons and then you should filter these events
# the lines to produce the Ref collection are the following
# you should not need to uncomment those, because I've already
# produced them in the ALCARECO step
#trackerDrivenOnlyElectrons = cms.EDFilter("GsfElectronRefSelector",
#                                          src = cms.InputTag( 'gedGsfElectrons' ),
#                                          cut = cms.string( "(ecalDrivenSeed==0)" )
#                                          )

# these lines active a filter that counts if there are more than 0
# trackerDrivenOnly electrons 
#trackerDrivenRemover = cms.EDFilter("PATCandViewCountFilter",
#                                    minNumber = cms.uint32(0),
#                                    maxNumber = cms.uint32(0),
#                                    src = cms.InputTag("trackerDrivenOnlyElectrons")
#                                    )
#trackerDrivenRemoverSeq = cms.Sequence( trackerDrivenOnlyElectrons * trackerDrivenRemover )
#trackerDrivenRemoverSeq = cms.Sequence( trackerDrivenOnlyElectrons)


# already imported by the WElectronSkim
#from RecoJets.Configuration.RecoPFJets_cff import *
#kt6PFJetsForRhoCorrection = kt6PFJets.clone(doRhoFastjet = True)
#kt6PFJetsForRhoCorrection.Rho_EtaMax = cms.double(2.5)




from Calibration.EcalAlCaRecoProducers.alCaIsolatedElectrons_cfi import *



seqALCARECOEcalCalElectronRECO = cms.Sequence( alCaIsolatedElectrons)


seqALCARECOEcalCalElectron = cms.Sequence( kt6PFJetsForRhoCorrection +
                                           seqALCARECOEcalCalElectronRECO
                                          )

seqALCARECOEcalCalZElectron = cms.Sequence( tagGsfSeq * seqALCARECOEcalCalElectron)
seqALCARECOEcalCalWElectron = cms.Sequence( WSkimSeq  * seqALCARECOEcalCalElectron) 

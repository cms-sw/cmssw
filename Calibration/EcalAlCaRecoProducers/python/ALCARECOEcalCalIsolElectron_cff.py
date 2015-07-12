import FWCore.ParameterSet.Config as cms


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


from Calibration.EcalAlCaRecoProducers.alCaIsolatedElectrons_cfi import *
from Calibration.EcalAlCaRecoProducers.AlCaElectronTracksReducer_cfi import *
from Calibration.EcalAlCaRecoProducers.eleIsoSequence_cff import *

from Calibration.EcalAlCaRecoProducers.WZElectronSkims_cff import *

from RecoJets.Configuration.RecoPFJets_cff import kt6PFJets
kt6PFJetsForRhoCorrection = kt6PFJets.clone(doRhoFastjet = True)
kt6PFJetsForRhoCorrection.Rho_EtaMax = cms.double(2.5)

#list of SCs to be used for the recHit reduction
#GsfMatchedPhotonCands = cms.EDProducer("ElectronMatchedCandidateProducer",
#   src     = cms.InputTag("goodPhotons"),
#   ReferenceElectronCollection = cms.untracked.InputTag("goodElectrons"),
#   deltaR =  cms.untracked.double(0.3)
#)

alcarecoEcalRecHitReducerSeq = cms.Sequence(alCaIsolatedElectrons)
alcarecoElectronTracksReducerSeq = cms.Sequence(alcaElectronTracksReducer)

# sequence that reduces the RECO format (only ECAL part) into ALCARECO
ALCARECOEcalCalElectronECALSeq = cms.Sequence( alCaIsolatedElectrons)

# sequence that reduces the RECO format (not ECAL part) into ALCARECO
ALCARECOEcalCalElectronNonECALSeq = cms.Sequence( kt6PFJetsForRhoCorrection +
                                              alcaElectronTracksReducer 
#                                             + pfisoALCARECO 
                                              )

ALCARECOEcalCalElectronSeq = cms.Sequence( ALCARECOEcalCalElectronNonECALSeq+
                                           ALCARECOEcalCalElectronECALSeq
                                           )
############################################### FINAL SEQUENCES
# sequences used in AlCaRecoStreams_cff.py
seqALCARECOEcalCalZElectron   = cms.Sequence(ZeeSkimFilterSeq  * ALCARECOEcalCalElectronSeq)
seqALCARECOEcalCalZSCElectron = cms.Sequence(ZSCSkimFilterSeq  * ALCARECOEcalCalElectronSeq)
seqALCARECOEcalCalWElectron   = cms.Sequence(WenuSkimFilterSeq * ALCARECOEcalCalElectronSeq)


seqALCARECOEcalCalPhoton = cms.Sequence( alCaIsolatedElectrons +
                                           kt6PFJetsForRhoCorrection 
                                         # + pfisoALCARECO 
                                         )




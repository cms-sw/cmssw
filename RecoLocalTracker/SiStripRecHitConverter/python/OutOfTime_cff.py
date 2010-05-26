import FWCore.ParameterSet.Config as cms

# ***lateFP is the percentage of the track missing closest to the front plane (strip side)
# ***lateBP is the percentage of the track missing closest to the back plane.

OutOfTime = cms.PSet( TIBlateFP = cms.double(0),
                      TIDlateFP = cms.double(0),
                      TOBlateFP = cms.double(0),
                      TEClateFP = cms.double(0),
                      TIBlateBP = cms.double(0),
                      TIDlateBP = cms.double(0),
                      TOBlateBP = cms.double(0),
                      TEClateBP = cms.double(0)
                      )

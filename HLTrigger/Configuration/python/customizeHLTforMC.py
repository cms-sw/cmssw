import FWCore.ParameterSet.Config as cms

def customizeHLTforMC(process):
  """adapt the HLT to run on MC, instead of data
  see Configuration/StandardSequences/Reconstruction_Data_cff.py
  which does the opposite, for RECO"""

  return process

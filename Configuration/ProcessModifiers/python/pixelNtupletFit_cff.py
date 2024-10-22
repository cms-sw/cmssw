import FWCore.ParameterSet.Config as cms

# This modifier is for replacing the legacy pixel tracks with the "Patatrack" pixel ntuplets,
# fishbone cleaning, and either the Broken Line fit (by default) or the Riemann fit.
# It also replaces the "gap" pixel vertices with a density-based vertex reconstruction algorithm.

pixelNtupletFit =  cms.Modifier()

import FWCore.ParameterSet.Config as cms

cleanMuonsBySegments = cms.EDProducer("MuonCleanerBySegments",
    # Collection of input muons
    src = cms.InputTag("muons"),
    # Preselection criteria: muons failing this are ignored completely
    preselection = cms.string("track.isNonnull"),
    # Pass-through definition: muons passing this are kept even if otherwise flagged as ghosts
    passthrough = cms.string("isGlobalMuon && numberOfMatches >= 2"),
    # Fraction of shared segments required for the muons to be considered overlapping
    fractionOfSharedSegments = cms.double(0.499), # i.e. clean up if sharing is >=50% 
    # Choice of the best muon: 
    #  - default: 
    #      1. prefer PF muons over non-PF ones
    #      2. prefer Global muons over non-Global ones
    #      3a. among split tracks, pick the one with best sigma(pt)/pt
    #      3b. among non-split tracks, pick the best by number of segments (or by pt in case of ties)
    #  - custom:
    #      a cut on a pair of muons which should be true if 'first' is better than 'second'
    #      an example that reproduces the default arbitration is below
    #customArbitration = cms.string("first.isPFMuon && !second.isPFMuon || "+   # <=== prefer PF Muons
    #                               "(first.isPFMuon == second.isPFMuon && ("+  #      over non-PF ones
    #                               "   first.isGlobalMuon && !second.isGlobalMuon || "+  # <=== prefer Global Muons
    #                               "   (first.isGlobalMuon == second.isGlobalMuon && "+  #      over non-Global ones
    #                               "   first.charge == second.charge && deltaR(first.eta,first.pt,second.eta,second.pt) < 0.03 && "+     #<=== Split Mu
    #                               "       first.track.ptError/first.track.pt < second.track.ptError/second.track.pt || "+ 
    #                               "   (first.charge != second.charge || deltaR(first.eta,first.pt,second.eta,second.pt) >= 0.03) && "+  #<=== Non-split Mu
    #                               "       (first.numberOfMatches('SegmentArbitration') >  second.numberOfMatches('SegmentArbitration') || "+
    #                               "        first.numberOfMatches('SegmentArbitration') == second.numberOfMatches('SegmentArbitration') && "+
    #                               "        first.pt > second.pt)"+
    #                               "   ))"+     # <=== end of prefer Global Muons
    #                               "   )")      # <=== end of prefer PF Muons
)


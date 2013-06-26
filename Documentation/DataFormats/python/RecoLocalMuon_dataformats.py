
full_title = "RecoLocalMuon collections (in RECO and AOD)"

full = {
    '0':['dt1DRecHits', '*', 'No documentation'] ,
    '1':['dt4DSegments', '*', 'No documentation'] ,
    '2':['csc2DRecHits', '*', 'No documentation'] ,
    '3':['cscSegments', '*', 'No documentation'] 
}

reco_title = "RecoLocalMuon collections (in RECO only)"

reco = {
    '0':['dt1DRecHits', 'DTRecHitCollection', 'DT 1D rechits with L/R ambiguity'] ,
    '1':['dt4DSegments', 'DTRecSegment4DCollection', 'DT segment in a full chamber'] ,
    '2':['dt1DCosmicRecHits', '*', 'No documentation'] ,
    '3':['dt4DCosmicSegments', '*', 'No documentation'] ,
    '4':['csc2DRecHits', 'CSCRecHit2DCollection', 'CSC rechits [A CSCRecHit2D is a reconstructed hit on one layer of a CSC. Effectively 3-dim: 2-d x, y but CSCLayer - labelled by CSCDetId - gives z]'] ,
    '5':['cscSegments', 'CSCSegmentCollection', 'CSC segments [A CSCSegment is built from the rechits in one CSC on a track; a CSCSegment is also, formally, itself a rechit]'], 

    # Correction needed, because not matched with Event Content
    '6':['DTSLRecSegment2D','DTRecSegment2DCollection','DT segment in one projection'],
    '7':['RPCRecHit','RPCRecHitCollection','RPC rechits']
}

aod_title = "RecoLocalMuon collections (in AOD only)"

aod = {

}
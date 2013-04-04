
full_title = "RecoLocalCalo collections (in RECO and AOD)"

full = {
    '0':['hbhereco', '*', 'No documentation'] ,
    '1':['hbheprereco', '*', 'No documentation'] ,
    '2':['hfreco', '*', 'No documentation'] ,
    '3':['horeco', '*', 'No documentation'] ,
    '4':['hbherecoMB', 'HBHERecHitsSorted', 'No documentation'] ,
    '5':['hbheprerecoMB', 'HBHERecHitsSorted', 'No documentation'] ,
    '6':['horecoMB', 'HORecHitsSorted', 'No documentation'] ,
    '7':['hfrecoMB', 'HFRecHitsSorted', 'No documentation'] ,
    '8':['*Digis', 'ZDCDataFramesSorted', 'No documentation'] ,
    '9':['*', 'ZDCRecHitsSorted', 'No documentation'] ,
    '10':['reducedHcalRecHits', '*', 'No documentation'] ,
    '11':['castorreco', '*', 'No documentation'] ,
    '12':['*', 'HcalUnpackerReport', 'No documentation'] 
}

reco_title = "RecoLocalCalo collections (in RECO only)"

reco = {
    '0':['hbhereco', 'edm::SortedCollection<HBHERecHit>', 'Joint HCAL barrel+endcap RecHits collection'] ,
    '1':['hfreco', 'edm::SortedCollection<HFRecHit>', 'Very Forward calorimeter RecHits collection'] ,
    '2':['horeco', 'edm::SortedCollection<HORecHit>', 'Outer clorimeter RecHits collection'] ,
    '3':['hbherecoMB', 'HBHERecHitsSorted', 'No documentation'] ,
    '4':['horecoMB', 'HORecHitsSorted', 'No documentation'] ,
    '5':['hfrecoMB', 'HFRecHitsSorted', 'No documentation'] ,
    '6':['*Digis', 'ZDCDataFramesSorted', 'No documentation'] ,
    '7':['zdcreco', 'ZDCRecHitsSorted', 'Zero-degree calorimeter RecHits collection'] ,
    '8':['reducedHcalRecHits', '*', 'No documentation'] ,
    '9':['castorreco', 'edm::SortedCollection<CastorRecHit>', 'Collection of CastorRecHits containing energy deposits for all channels'] ,
    '10':['*', 'HcalUnpackerReport', 'No documentation'],
    
    # Correction needed, because not matched with Event Content
    '11':['ecalRecHit,EcalRecHitsEB', 'edm::SortedCollection<EcalRecHit>', 'Collection of Ecal Hits in EB'],
    '12':['ecalRecHit,EcalRecHitsEE', 'edm::SortedCollection<EcalRecHit>', 'Collection of Ecal Hits in EE'],
    '13':['ecalPreshowerRecHit,EcalRecHitsES', 'edm::SortedCollection<EcalRecHit>', 'Collection of Ecal Hits in ES'] 
}

aod_title = "RecoLocalCalo collections (in AOD only)"

aod = {
    '0':['castorreco', '*', 'No documentation'] ,
    '1':['reducedHcalRecHits', '*', 'No documentation'] ,
    '2':['*', 'HcalUnpackerReport', 'No documentation'] 
}
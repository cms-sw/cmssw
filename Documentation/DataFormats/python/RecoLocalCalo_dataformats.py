'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoLocalCalo collections (in RECO and AOD)",
    "data": [
     {
      "instance": "castorreco",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "reducedHcalRecHits",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "*",
      "container": "HcalUnpackerReport",
      "desc": "No documentation"
     },
     {
      "instance": "hbheprereco",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hbhereco",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "horeco",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hfreco",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "hbheprerecoMB",
      "container": "HBHERecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "hbherecoMB",
      "container": "HBHERecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "hfrecoMB",
      "container": "HFRecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "horecoMB",
      "container": "HORecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "*",
      "container": "ZDCRecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "*Digis",
      "container": "ZDCDataFramesSorted",
      "desc": "No documentation"
     }
    ]
  },
  "aod": {
    "title": "RecoLocalCalo collections (in AOD only)",
    "data": [
     {
      "instance": "reducedHcalRecHits",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "castorreco",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "*",
      "container": "HcalUnpackerReport",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "RecoLocalCalo collections (in RECO only)",
    "data": [
     {
      "instance": "ecalRecHit,EcalRecHitsEB",
      "container": "edm::SortedCollection<EcalRecHit>",
      "desc": "Collection of Ecal Hits in EB"
     },
     {
      "instance": "*",
      "container": "HcalUnpackerReport",
      "desc": "No documentation"
     },
     {
      "instance": "ecalPreshowerRecHit,EcalRecHitsES",
      "container": "edm::SortedCollection<EcalRecHit>",
      "desc": "Collection of Ecal Hits in ES"
     },
     {
      "instance": "ecalRecHit,EcalRecHitsEE",
      "container": "edm::SortedCollection<EcalRecHit>",
      "desc": "Collection of Ecal Hits in EE"
     },
     {
      "instance": "hfreco",
      "container": "edm::SortedCollection<HFRecHit>",
      "desc": "Very Forward calorimeter RecHits collection"
     },
     {
      "instance": "hbhereco",
      "container": "edm::SortedCollection<HBHERecHit>",
      "desc": "Joint HCAL barrel+endcap RecHits collection"
     },
     {
      "instance": "hbherecoMB",
      "container": "HBHERecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "horeco",
      "container": "edm::SortedCollection<HORecHit>",
      "desc": "Outer clorimeter RecHits collection"
     },
     {
      "instance": "hfrecoMB",
      "container": "HFRecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "horecoMB",
      "container": "HORecHitsSorted",
      "desc": "No documentation"
     },
     {
      "instance": "zdcreco",
      "container": "ZDCRecHitsSorted",
      "desc": "Zero-degree calorimeter RecHits collection"
     },
     {
      "instance": "*Digis",
      "container": "ZDCDataFramesSorted",
      "desc": "No documentation"
     },
     {
      "instance": "castorreco",
      "container": "edm::SortedCollection<CastorRecHit>",
      "desc": "Collection of CastorRecHits containing energy deposits for all channels"
     },
     {
      "instance": "reducedHcalRecHits",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  }
}

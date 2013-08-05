'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "L1Trigger collections (in RECO and AOD)",
    "data": [

    ]
  },
  "aod": {
    "title": "L1Trigger collections (in AOD only)",
    "data": [
     {
      "instance": "l1GtRecord",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "gtDigis",
      "container": "L1GlobalTriggerReadoutRecord",
      "desc": "No documentation"
     },
     {
      "instance": "conditionsInEdm",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "l1GtTriggerMenuLite",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "l1L1GtObjectMap",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "l1extraParticles",
      "container": "*",
      "desc": "No documentation"
     }
    ]
  },
  "reco": {
    "title": "L1Trigger collections (in RECO only)",
    "data": [
     {
      "instance": "gctDigis",
      "container": "L1GctEtTotal*",
      "desc": "No documentation"
     },
     {
      "instance": "gctDigis",
      "container": "L1GctEtMiss*",
      "desc": "No documentation"
     },
     {
      "instance": "gctDigis",
      "container": "L1GctJetCounts*",
      "desc": "No documentation"
     },
     {
      "instance": "gctDigis",
      "container": "L1GctHtMiss*",
      "desc": "No documentation"
     },
     {
      "instance": "gctDigis",
      "container": "L1GctHFBitCounts*",
      "desc": "No documentation"
     },
     {
      "instance": "gctDigis",
      "container": "L1GctHFRingEtSums*",
      "desc": "No documentation"
     },
     {
      "instance": "lumiProducer",
      "container": "LumiDetails",
      "desc": "No documentation"
     },
     {
      "instance": "l1GtRecord",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "gtDigis",
      "container": "L1GlobalTriggerReadoutRecord",
      "desc": "No documentation"
     },
     {
      "instance": "conditionsInEdm",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "l1GtTriggerMenuLite",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "l1L1GtObjectMap",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "l1extraParticles",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "gctDigis",
      "container": "L1GctEmCand*",
      "desc": "No documentation"
     },
     {
      "instance": "gtDigis",
      "container": "L1MuGMTReadoutCollection",
      "desc": "No documentation"
     },
     {
      "instance": "gctDigis",
      "container": "L1GctEtHad*",
      "desc": "No documentation"
     },
     {
      "instance": "gctDigis",
      "container": "L1GctJetCand*",
      "desc": "No documentation"
     }
    ]
  }
}

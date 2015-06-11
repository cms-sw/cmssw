autoSkim = {
    'Commissioning':'DT+LogError',
    'Cosmics':'CosmicSP+CosmicTP+LogError',
    'CosmicsSP':'CosmicSP+LogError',
    'TopMuEG':'TopMuEG+LogError',
    'ZElectron' : 'ZElectron+LogError',
    'ZMu' : 'ZMu+LogError',
    'HighMET': 'HighMET+LogError',
    'MuTau': 'MuTau+LogError',
    
    }


autoSkimPDWG = {
    
    }

autoSkimDPG = {

    }

def mergeMapping(map1,map2):
    merged={}
    for k in list(set(map1.keys()+map2.keys())):
        items=[]
        if k in map1: 
            items.append(map1[k])
        if k in map2:
            items.append(map2[k])
        merged[k]='+'.join(items)
    return merged
    
#autoSkim = mergeMapping(autoSkimPDWG,autoSkimDPG)
#print autoSkim

"""
autoPAT - Automatic PAT/MiniAOD and NanoAOD configuration mappings for scouting.

Defines flavors for PAT and NANO steps:
    --step PAT:@Scout   (HLTSCOUT -> MiniAOD)
    --step NANO:@ScoutMini  (Scouting MiniAOD -> NanoAOD)

Usage:
    cmsDriver.py ... --step PAT:@Scout ...
    cmsDriver.py ... --step NANO:@ScoutMini ...
"""

def expandPATMapping(seqList, mapping, key):
    """Expand @-prefixed mappings in seqList using the mapping dictionary."""
    maxLevel = 30
    level = 0
    while '@' in repr(seqList) and level < maxLevel:
        level += 1
        for specifiedCommand in seqList:
            if specifiedCommand.startswith('@'):
                location = specifiedCommand[1:]
                if not location in mapping:
                    raise Exception("Impossible to map " + location + " from " + repr(mapping))
                mappedTo = mapping[location]
                insertAt = seqList.index(specifiedCommand)
                seqList.remove(specifiedCommand)
                if key in mappedTo and mappedTo[key] is not None:
                    allToInsert = mappedTo[key].split('+')
                    for offset, toInsert in enumerate(allToInsert):
                        seqList.insert(insertAt + offset, toInsert)
                break
        if level == maxLevel:
            raise Exception("Could not fully expand " + repr(seqList) + " from " + repr(mapping))


# PAT flavor definitions for --step PAT:@...
autoPAT = {
    # Standard MiniAOD (default)
    'PHYS': {'sequence': '',
             'customize': ''},

    # Scouting MiniAOD from HLTSCOUT input
    'Scout': {
        'sequence': 'PhysicsTools/PatFromScouting/scoutingToMiniAOD_cff.scoutingToMiniAODTask',
        'customize': 'PhysicsTools/PatFromScouting/scoutingToMiniAOD_cff.customiseScoutingToMiniAOD'
    },

    # Scouting MiniAOD with vertex muons (2024+)
    'ScoutVtx': {
        'sequence': '@Scout',
        'customize': '@Scout+PhysicsTools/PatFromScouting/scoutingToMiniAOD_cff.customiseScoutingToMiniAOD_withMuonVtx'
    },
}


# NANO flavor definitions for --step NANO:@...
# These extend the central autoNANO with scouting-specific flavors
autoNANO_scouting = {
    # NanoAOD from Scouting MiniAOD (PAT objects)
    # This reads from slimmedMuons, slimmedJets, etc. produced by PAT:@Scout
    'ScoutMini': {
        'sequence': 'PhysicsTools/PatFromScouting/scoutingNanoAOD_cff.scoutingNanoAODTask',
        'customize': 'PhysicsTools/PatFromScouting/scoutingNanoAOD_cff.customiseScoutingNanoAOD'
    },
}

def expandNanoMapping(seqList, mapping, key):
    maxLevel=30
    level=0
    while '@' in repr(seqList) and level<maxLevel:
        level+=1
        for specifiedCommand in seqList:
            if specifiedCommand.startswith('@'):
                location=specifiedCommand[1:]
                if not location in mapping:
                    raise Exception("Impossible to map "+location+" from "+repr(mapping))
                mappedTo=mapping[location]
                # no mapping for specified key
                # NOTE: mising key of key=None is interpreted differently than empty string:
                #  - An empty string recalls the default for the given key
                #  - None is interpreted as "ignore this"
                insertAt=seqList.index(specifiedCommand)
                seqList.remove(specifiedCommand)
                if key in mappedTo and mappedTo[key] is not None:
                    allToInsert=mappedTo[key].split('+')
                    for offset,toInsert in enumerate(allToInsert):
                        seqList.insert(insertAt+offset,toInsert)
                break;
        if level==maxLevel:
            raise Exception("Could not fully expand "+repr(seqList)+" from "+repr(mapping))


autoNANO = {
    # PHYS is a mapping to the default NANO config, i.e. empty strings
    'PHYS': {'sequence': '',
             'customize': ''},
    # L1 flavours: add tables through customize, supposed to be combined with PHYS
    'L1' : {'customize': 'nanoL1TrigObjCustomize'},
    'L1FULL' : {'customize': 'nanoL1TrigObjCustomizeFull'},
    #scouting nano
    'Scout' : {'sequence': 'PhysicsTools/NanoAOD/custom_run3scouting_cff'},
    # JME custom NANO
    'JME' : { 'sequence': '@PHYS',
               'customize': '@PHYS+PhysicsTools/NanoAOD/custom_jme_cff.PrepJMECustomNanoAOD'},
    'JMErePuppi' : { 'sequence': '@PHYS',
                     'customize': '@PHYS+@JME+PhysicsTools/NanoAOD/custom_jme_cff.RecomputePuppiWeightsAndMET'},
    # Muon POG flavours : add tables through customize, supposed to be combined with PHYS
    'MUPOG' : { 'sequence': '@PHYS',
                'customize' : '@PHYS+PhysicsTools/NanoAOD/custom_muon_cff.PrepMuonCustomNanoAOD'},
    # MUDPG flavours: use their own sequence
    'MUDPG' : {'sequence': 'DPGAnalysis/MuonTools/muNtupleProducer_cff.muDPGNanoProducer',
               'customize': 'DPGAnalysis/MuonTools/muNtupleProducer_cff.muDPGNanoCustomize'},
    'MUDPGBKG' : {'sequence': 'DPGAnalysis/MuonTools/muNtupleProducerBkg_cff.muDPGNanoProducerBkg',
                  'customize': 'DPGAnalysis/MuonTools/muNtupleProducerBkg_cff.muDPGNanoBkgCustomize'},
    #EGM flavours: add variables through customize
    'EGM' : {'sequence': '@PHYS',
             'customize' : '@PHYS+PhysicsTools/NanoAOD/egamma_custom_cff.addExtraEGammaVarsCustomize'},
    # PromptReco config: PHYS+L1
    'Prompt' : {'sequence': '@PHYS',
                'customize': '@PHYS+@L1'},
    # Add lepton track parameters through customize combined with PHYS
    'LepTrackInfo' : {'sequence': '@PHYS',
                      'customize': '@PHYS+PhysicsTools/NanoAOD/leptonTimeLifeInfo_common_cff.addTrackVarsToTimeLifeInfo'},
    # Custom BTV Nano for SF measurements or tagger training
    'BTV' : {'sequence': '@PHYS',
            'customize':'@PHYS+PhysicsTools/NanoAOD/custom_btv_cff.BTVCustomNanoAOD'},
    # Tau embedding NanoAOD (includes Data as well as MC tables with an addition tau embedding table)
    'TauEmbedding': {'sequence': '@PHYS+PhysicsTools/NanoAOD/nano_cff.nanoSequenceFS+TauAnalysis/MCEmbeddingTools/Nano_cff.embeddingTable_seq',
                     'customize': '@PHYS'},
}

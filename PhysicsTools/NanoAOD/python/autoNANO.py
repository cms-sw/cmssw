def expandNanoMapping(seqList, mapping, key):
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
                # no mapping for specified key
                # NOTE: mising key of key=None is interpreted differently than empty string:
                #  - An empty string recalls the default for the given key
                #  - None is interpreted as "ignore this"
                insertAt = seqList.index(specifiedCommand)
                seqList.remove(specifiedCommand)
                if key in mappedTo and mappedTo[key] is not None:
                    allToInsert = mappedTo[key].split('+')
                    for offset, toInsert in enumerate(allToInsert):
                        seqList.insert(insertAt + offset, toInsert)
                break
        if level == maxLevel:
            raise Exception("Could not fully expand " + repr(seqList) + " from " + repr(mapping))


autoNANO = {
    # PHYS is a mapping to the default NANO config, i.e. empty strings
    'PHYS': {'sequence': '',
             'customize': ''},
    # L1 flavours: add tables through customize, supposed to be combined with PHYS
    'L1': {'customize': 'PhysicsTools/NanoAOD/l1trig_cff.nanoL1TrigObjCustomize'},
    'L1FULL': {'customize': 'PhysicsTools/NanoAOD/l1trig_cff.nanoL1TrigObjCustomizeFull'},
    # scouting nano
    'Scout': {'sequence': 'PhysicsTools/NanoAOD/custom_run3scouting_cff'},
    # JME nano
    'JME': {'sequence': '@PHYS',
            'customize': '@PHYS+PhysicsTools/NanoAOD/custom_jme_cff.PrepJMECustomNanoAOD'},
    'JMErePuppi': {'sequence': '@PHYS',
                   'customize': '@PHYS+@JME+PhysicsTools/NanoAOD/custom_jme_cff.RecomputePuppiWeightsMETAK8'},
    # L1 DPG (standalone with full calo TP info, L1T reemulation customization)
    'L1DPG' : {'sequence': 'DPGAnalysis/L1TNanoAOD/l1tNano_cff.l1tNanoSequence',
               'customize': ','.join(['PhysicsTools/NanoAOD/l1trig_cff.nanoL1TrigObjCustomizeFull',
                                      'DPGAnalysis/L1TNanoAOD/l1tNano_cff.addCaloFull',
                                      'L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW'])},
    # Muon POG flavours : add tables through customize, supposed to be combined with PHYS
    'MUPOG': {'sequence': '@PHYS',
              'customize': '@PHYS+PhysicsTools/NanoAOD/custom_muon_cff.PrepMuonCustomNanoAOD'},
    # MUDPG flavours: use their own sequence
    'MUDPG': {'sequence': 'DPGAnalysis/MuonTools/muNtupleProducer_cff.muDPGNanoProducer',
              'customize': 'DPGAnalysis/MuonTools/muNtupleProducer_cff.muDPGNanoCustomize'},
    'MUDPGBKG': {'sequence': 'DPGAnalysis/MuonTools/muNtupleProducerBkg_cff.muDPGNanoProducerBkg',
                 'customize': 'DPGAnalysis/MuonTools/muNtupleProducerBkg_cff.muDPGNanoBkgCustomize'},
    # HCAL flavors:
    'HCAL': {'sequence': 'DPGAnalysis/HcalNanoAOD/hcalNano_cff.hcalNanoTask'},
    'HCALCalib': {'sequence': 'DPGAnalysis/HcalNanoAOD/hcalNano_cff.hcalNanoTask',
                  'customize': 'DPGAnalysis/HcalNanoAOD/customiseHcalCalib_cff.customiseHcalCalib'},
    # EGM flavours: add variables through customize
    'EGM': {'sequence': '@PHYS',
            'customize': '@PHYS+PhysicsTools/NanoAOD/egamma_custom_cff.addExtraEGammaVarsCustomize'},
    # PromptReco config: PHYS+L1
    'Prompt': {'sequence': '@PHYS',
               'customize': '@PHYS+@L1'},
    # Add lepton track parameters through customize combined with PHYS
    'LepTrackInfo' : {'sequence': '@PHYS',
                      'customize': '@PHYS+PhysicsTools/NanoAOD/leptonTimeLifeInfo_common_cff.addTrackVarsToTimeLifeInfo'},
    # Custom BTV Nano for SF measurements or tagger training
    'BTV': {'sequence': '@PHYS',
            'customize': '@PHYS+PhysicsTools/NanoAOD/custom_btv_cff.BTVCustomNanoAOD'},
    # NANOGEN (from LHE/GEN/AOD)
    'GEN': {'sequence': 'PhysicsTools/NanoAOD/nanogen_cff.nanogenSequence',
            'customize': 'PhysicsTools/NanoAOD/nanogen_cff.customizeNanoGEN'},
    # NANOGEN (from MiniAOD)
    'GENFromMini': {'sequence': 'PhysicsTools/NanoAOD/nanogen_cff.nanogenSequence',
                    'customize': 'PhysicsTools/NanoAOD/nanogen_cff.customizeNanoGENFromMini'},
}

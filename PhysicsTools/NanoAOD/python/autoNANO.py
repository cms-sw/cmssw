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
                seqList.remove(specifiedCommand)
                if key in mappedTo and mappedTo[key] is not None:                    
                    seqList.extend(mappedTo[key].split('+'))
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
    # PromptReco config: PHYS+L1
    'Prompt' : {'sequence': '@PHYS', 
                'customize': '@PHYS+@L1'}
}

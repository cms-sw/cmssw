#!/bin/python3

import re

def getMixture(line) :
    mixture = re.search(
        r'^#\s*"(?P<MixtureName>[^"]+)"\s+"(?P<GMIXName>[^"]+)"\s+(?P<MCVolume>[0-9.-]+)\s+(?P<MCArea>[0-9.-]+)',
        line
    )
    return {
        'mixture_name' : mixture.group("MixtureName"),
        'gmix_name' : mixture.group("GMIXName"),
        'mc_volume' : float(mixture.group("MCVolume")),
        'mc_area' : float(mixture.group("MCArea")),
    }


def getCompound(line) :
    compound = re.search(
        r'^\*\s*(?P<Index>[0-9]+)\s+"(?P<Comment>[^"]+)"\s+"(?P<Material>[^"]+)"\s+(?P<Volume>[0-9.]+)\s+(?P<Mult>[0-9.]+)\s+(?P<Type>[^ ]{3})',
        line
    )
    return {
        'item_number' : int(compound.group("Index")),
        'comment' : compound.group("Comment"),
        'material' : compound.group("Material"),
        'volume' : float(compound.group("Volume")),
        'multiplicity' : float(compound.group("Mult")),
        'category' : compound.group("Type"),
    }

def getMixtures() :

    """
    Returns a list of Mixtures (dict). Each
    mixture contains a list of compounds.
    """

    inFile = open("pixel_fwd.in","r")

    mixtures = []
    mixture = {}

    counter = 0
    prevCounter = 0

    for line in inFile.readlines():

        if line[0] == "#":
            if mixture:
                mixtures.append(dict(mixture)) #Just to make a copy
            mixture.clear()
            mixture = getMixture(line)
            mixture.update({ 'compounds' : [] })

        if line[0] == "*":
            mixture['compounds'].append(dict(getCompound(line)))

    inFile.close()

    return mixtures

#!/bin/python3

import re
import pandas as pd

def getMixture(line) :
    mixture = re.search(
        r'^#\s*"(?P<MixtureName>[^"]+)"\s+"(?P<GMIXName>[^"]+)"\s+(?P<MCVolume>[0-9.Ee-]+)\s+(?P<MCArea>[0-9.Ee-]+)',
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
        r'^\*\s*(?P<Index>[0-9]+)\s+"(?P<Comment>[^"]+)"\s+"(?P<Material>[^"]+)"\s+(?P<Volume>[0-9.Ee-]+)\s+(?P<Mult>[0-9.]+)\s+(?P<Type>[^ ]{3})',
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

    inFile = open("pixel_bar.in","r")

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
            mixture.update({ 'components' : [] })

        if line[0] == "*":
            mixture['components'].append(dict(getCompound(line)))

    inFile.close()

    return mixtures

def loadMaterialsFile(inputFile):

    mixMatFile = open(inputFile,"r")

    mixMats = []

    for line in mixMatFile.readlines():

        if line[0] == '"':
            mixMat = re.search(
                r'^"(?P<name>[^"]+)"\s+(?P<weight>[0-9.Ee-]+)\s+(?P<number>[0-9.Ee-]+)\s+(?P<density>[0-9.Ee-]+)\s+(?P<x0>[0-9.Ee-]+)\s+(?P<l0>[0-9Ee.-]+)',
                line
                )

            mixMats.append({
                    "name" : mixMat.group("name"),
                    "weight" : float(mixMat.group("weight")),
                    "number" : float(mixMat.group("number")),
                    "density" : float(mixMat.group("density")),
                    "x0" : float(mixMat.group("x0")),
                    "l0" : float(mixMat.group("l0")),
                    })

    return mixMats

listOfMixtures = getMixtures()

listOfMixedMaterials = loadMaterialsFile(inputFile="mixed_materials.input")
listOfPureMaterials = loadMaterialsFile(inputFile="pure_materials.input")
listOfMaterials = listOfMixedMaterials + listOfPureMaterials


dfAllMaterials = pd.DataFrame(listOfMaterials)


for index in range(len(listOfMixtures)):
    print("================================")
    print(listOfMixtures[index]["gmix_name"])


    components = pd.DataFrame(listOfMixtures[index]['components'])

    components["volume"] = components["multiplicity"]*components["volume"]
    totalVolume = sum(
        components["volume"]
        )

    components["percentVolume"] = components["volume"] / totalVolume

    components = pd.merge(
        components,
        dfAllMaterials[["name","density","x0","l0"]],
        left_on="material",
        right_on="name"
        )

    print(components)
    components["weight"] = components["density"] * components["volume"]

    totalDensity = sum(
        components["percentVolume"] * components["density"]
        )

    totalWeight = totalDensity * totalVolume

    components["percentWeight"] = (
        components["density"] * components["volume"] / totalWeight
        )

    components["ws"] = (
        components["percentWeight"] / (components["density"]*components["x0"])
        )

    components["ws2"] = (
        components["percentWeight"] / (components["density"]*components["l0"])
        )

    totalRadLen = 1 / ( sum(components["ws"]) * totalDensity )

    totalIntLen = 1/ ( sum(components["ws2"]) * totalDensity )

    components["percentRadLen"] = (
        components["ws"] * totalRadLen * totalDensity
        )

    components["percentIntLen"] = (
        components["ws2"] * totalIntLen * totalDensity
        )


    print("Mixture Density:" , totalDensity)
    print("Mixture Volume:", totalVolume)
    print("Mixture x0:",totalRadLen)
    print("Mixture l0:",totalIntLen)
    print("Total Weight:",totalWeight)

    print(components[[
                "name","material","volume",
                "percentVolume","percentWeight",
                "density","weight","x0","percentRadLen",
                "l0","percentRadLen"
                ]])

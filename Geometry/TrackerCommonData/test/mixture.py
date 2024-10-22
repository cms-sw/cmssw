#!/usr/bin/env python3

"""
mixture.py replaces the functionality of mixture.f

Usage:

$ python3 mixture.py filename.in

Where `filename.in` contains a set of mixtures and its components:

The format for `filename.in` is as follows:

Each mixture description starts with a hash symbol (#) and contains
the following information in a single line:

 * Name (name of mixture of components)
 * GMIX name ()
 * MC Volume
 * MC Area

A mixture definition is followed by a set of compounds, which is the
set of components that the mixture is made from.

Each compound description starts with an asterisk (*) and contains
the following information:

 * Item number (An index for the list)
 * Comment (A human readable name for the material)
 * Material (Component name)
 * Volume
 * Multiplicity (How many times the material is in the mixture)
 * Type (see below)

Type stands for a classification of the Material using the following
convention:

 * SUP: Support
 * COL: Cooling
 * ELE: Electronics
 * CAB: Cables
 * SEN: Sensitive volumes

Each Material (component of the mixture) belongs to a single category.
`mixture.py` uses these information to compute how the mixture as a whole
contributes to each of the categories in terms of radiation length (x0)
and nuclear interaction length (l0) \lambda_{0}.

The radiation length or the nuclear interaction lenght become relevant
depending if we are talking about particles that interact with the detector
mainly throgh electromagnetic forces (x0) or either strong interaction or nuclear
forces (l0). Notice that these quantities are different and therefore the
contribution for each category will depend on them.

Any line in `filename.in` is therefore ignored unless it begins with
a hash or an asterisk.

Hardcoded in `mixture.py` are the files `mixed_materials.input` and
`pure_materials.input`. These files act as a database that provides
information about the compounds. From each compound defined in
`filename.in` `mixture.py` matches the content of `Material` with a
single element of the database and extracts the following information
(the format of the mentioned .input files):

 * Name (which should match with Material)
 * Weight (Atomic Mass) [g/mol]
 * Number (Atomic number)
 * Density (g/cm^3)
 * RadLen (Radiation lenght) (cm)
 * IntLen (Nuclear interaction lenght) (cm)

With these information `material.py` computes for each mixture:

 * The radiation lenght for the mixture
 * The nuclear interaction lenght for the mixture
 * The mixture density
 * The mixture volume
 * The total weight
 * The relative contribution in each categorio on x0 and l0
 * Normalized quantities based on MC Volume and MC Area values

As well for the compounds:

 * The relative volume within the mixture
 * The relative weight
 * The relative radiation lenght
 * The relative nuclear interaction length

"""

import re
import pandas as pd
import sys
from os.path import expandvars

def getMixture(line) :
    mixture = re.search(
        r'^#\s*"(?P<MixtureName>[^"]+)"\s+"(?P<GMIXName>[^"]+)"\s+(?P<MCVolume>[0-9.Ee\-+]+)\s+(?P<MCArea>[0-9.Ee\-+]+)',
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
        r'^\*\s*(?P<Index>[0-9]+)\s+"(?P<Comment>[^"]+)"\s+"(?P<Material>[^"]+)"\s+(?P<Volume>[0-9.Ee\-+]+)\s+(?P<Mult>[0-9.]+)\s+(?P<Type>[^ ]{3})',
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

def getMixtures(inputFile) :

    """
    Returns a `dict` of Mixtures. Each
    mixture contains a list of compounds.
    """

    inFile = open(inputFile,"r")

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

def main(argv):

    listOfMixtures = getMixtures(str(argv[1]))

    fname = re.search("(?P<filename>[a-zA-Z0-9_]+)",str(argv[1]))

    fname = str(fname.group("filename"))
    radLenFile = open(fname + ".x0","w");
    intLenFile = open(fname + ".l0","w");

    listOfMixedMaterials = loadMaterialsFile(
        inputFile = expandvars(
            "$CMSSW_BASE/src/Geometry/TrackerCommonData/data/Materials/mixed_materials.input"
        )
    )
    listOfPureMaterials = loadMaterialsFile(
        inputFile = expandvars(
            "$CMSSW_BASE/src/Geometry/TrackerCommonData/data/Materials/pure_materials.input"
        )
    )
    listOfMaterials = listOfMixedMaterials + listOfPureMaterials

    dfAllMaterials = pd.DataFrame(listOfMaterials)

    for index in range(len(listOfMixtures)):

        gmixName = listOfMixtures[index]["gmix_name"]
        print("================================")
        print(gmixName)

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

        displayDf = components[[
                    "comment",
                    "material","volume",
                    "percentVolume","percentWeight",
                    "density","weight","x0","percentRadLen",
                    "l0","percentIntLen"
                    ]].copy()

        displayDf["percentVolume"] *= 100.
        displayDf["percentWeight"] *= 100.
        displayDf["percentRadLen"] *= 100.
        displayDf["percentIntLen"] *= 100.

        displayDf = displayDf.rename(
            columns = {
                "comment" : "Component",
                "material" : "Material",
                "volume" : "Volume [cm^3]",
                "percentVolume" : "Volume %",
                "density" : "Density",
                "weight" : "Weight",
                "percentWeight" : "Weight %",
                "x0" : "X_0 [cm]",
                "percentRadLen" : "X_0 %",
                "l0": "lambda_0 [cm]",
                "percentIntLen" : "lambda_0 %",
                }
            )

        print(displayDf)


        # Normalized vars:

        mcVolume = listOfMixtures[index]["mc_volume"]
        mcArea = listOfMixtures[index]["mc_area"]
        normalizationFactor = -1.

        if mcVolume > 0:
            normalizationFactor = totalVolume / mcVolume
        else:
            normalizationFactor = 1.

        normalizedDensity = totalDensity * normalizationFactor
        normalizedRadLen = totalRadLen / normalizationFactor
        normalizedIntLen = totalIntLen / normalizationFactor

        percentRadL = -1.
        percentIntL = -1.

        if mcArea > 0:
            percentRadL = mcVolume / (mcArea*normalizedRadLen)
            percentIntL = mcVolume / (mcArea*normalizedIntLen)

        pSupRadLen = 0.
        pSenRadLen = 0.
        pCabRadLen = 0.
        pColRadLen = 0.
        pEleRadLen = 0.

        pSupIntLen = 0.
        pSenIntLen = 0.
        pCabIntLen = 0.
        pColIntLen = 0.
        pEleIntLen = 0.

        for index, component in components.iterrows():
            catg = component["category"]
            prl = component["percentRadLen"]
            irl = component["percentIntLen"]
            if catg.upper() == "SUP":
                pSupRadLen += prl
                pSupIntLen += irl
            elif catg.upper() == "SEN":
                pSenRadLen += prl
                pSenIntLen += irl
            elif catg.upper() == "CAB":
                pCabRadLen += prl
                pCabIntLen += irl
            elif catg.upper() == "COL":
                pColRadLen += prl
                pColIntLen += irl
            elif catg.upper() == "ELE":
                pEleRadLen += prl
                pEleIntLen += irl

        print("================================")
        print("Mixture Density       [g/cm^3]: " , totalDensity)
        print("Norm. mixture density [g/cm^3]: ", normalizedDensity)
        print("Mixture Volume        [cm^3]: ", totalVolume)
        print("MC Volume             [cm^3]: ", mcVolume)
        print("MC Area               [cm^2]: ", mcArea)
        print("Normalization Factor:       ",  normalizationFactor)
        print("Mixture x0            [cm]: ", totalRadLen)
        print("Norm. Mixture x0      [cm]: ", normalizedRadLen)
        if mcArea > 0 :
            print("Norm. Mixture x0      [%]: ", 100. * percentRadL)
        print("Mixture l0            [cm]: ", totalIntLen)
        print("Norm. Mixture l0      [cm]: ", normalizedIntLen)
        if mcArea > 0 :
            print("Norm. Mixture l0      [%]: ", 100. * percentIntL)
        print("Total Weight          [g]: ", totalWeight)

        print("================================")
        print("X0 Contribution: ")
        print("Support     :", pSupRadLen)
        print("Sensitive   :", pSenRadLen)
        print("Cables      :", pCabRadLen)
        print("Cooling     :", pColRadLen)
        print("Electronics :", pEleRadLen)

        print("================================")
        print("l0 Contribution: ")
        print("Support     :", pSupIntLen)
        print("Sensitive   :", pSenIntLen)
        print("Cables      :", pCabIntLen)
        print("Cooling     :", pColIntLen)
        print("Electronics :", pEleIntLen)

        radLenFile.write("{0:<50}{1:>8}{2:>8}{3:>8}{4:>8}{5:>8}\n".format(
                gmixName,
                round(pSupRadLen,3), round(pSenRadLen,3),
                round(pCabRadLen,3), round(pColRadLen,3),
                round(pEleRadLen,3)
                ))

        intLenFile.write("{0:<50}{1:>8}{2:>8}{3:>8}{4:>8}{5:>8}\n".format(
                gmixName,
                round(pSupIntLen,3), round(pSenIntLen,3),
                round(pCabIntLen,3), round(pColIntLen,3),
                round(pEleIntLen,3)
                ))

    radLenFile.close()
    intLenFile.close()

if __name__== "__main__":
    main(sys.argv)

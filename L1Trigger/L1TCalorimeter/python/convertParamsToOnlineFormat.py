#!/bin/env python

from __future__ import print_function
import argparse
import FWCore.ParameterSet.Config as cms
from importlib import import_module
import os
import sys
import xml.etree.ElementTree as ET

import six

# Pairwise generator: returns pairs of adjacent elements in a list / other iterable
def pairwiseGen(aList):
    for i in xrange(len(aList)-1):
        yield (aList[i], aList[i+1])

def parseOfflineLUTfile(aRelPath, aExpectedSize, aPaddingValue = None, aTruncate = False):
    # Find file by looking under directories listed in 'CMSSW_SEARCH_PATH' as outlined in https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEdmFileInPath
    searchPaths = os.getenv('CMSSW_SEARCH_PATH').split(':')
    resolvedPath = None
    for baseDir in searchPaths:
        print("Looking for '" + aRelPath + "' under '" + baseDir + "'")
        if os.path.isfile(os.path.join(baseDir, aRelPath)):
            print("   success!")
            resolvedPath = os.path.join(baseDir, aRelPath)
            break
    if resolvedPath is None:
        raise RuntimeError("Could not find LUT file '" + aRelPath + "' under directories in 'CMSSW_SEARCH_PATH'")

    with open(resolvedPath) as f:
        entries = []
        line_nr = 0
        for line in f:
            line_nr += 1
            # Ignore comment lines
            if line.startswith('#') or line == '\n':
                continue

            # Remove trailing comments from data lines
            stripped_line = line[:line.find('#')]

            # Split line into list of whitespace-separated items
            items = stripped_line.split()
            if len(items) != 2:
               print("ERROR parsing file", resolvedPath, "on line", line_nr, "'" + line + "' : Splitting on whitespace produced", len(items), "items")
               sys.exit(1)

            entries.append( (int(items[0]), int(items[1])) )

        # Sort the LUT
        entries.sort(key= lambda x : x[0])  
        # Check that the LUT is not empty
        if len(entries) == 0:
            print("ERROR parsing file", resolvedPath, ": No LUT entries defined in the file")
            sys.exit(1)

        # Check that no items from the LUT are missing
        if entries[0][0] != 0:
            print("ERROR parsing file", resolvedPath, ": LUT entries before index", entries[0][0], "are not defined")
            sys.exit(1)
         
        for x1, x2 in pairwiseGen(entries):
            if x1[0] != (x2[0]-1):
                print("ERROR parsing file", resolvedPath, ": ", x2[0] - x1[0] - 1,"LUT entries between indices", x1[0], "and", x2[0], "are not defined")
                sys.exit(1)

        result = [x[1] for x in entries]

        if (len(result) < aExpectedSize) and not (aPaddingValue is None):
            print ("WARNING : Padding", str(len(result))+"-entry LUT with value", aPaddingValue, "to have", aExpectedSize, "entries")
            result += ([aPaddingValue] * (aExpectedSize - len(result)))
        elif (len(result) > aExpectedSize) and aTruncate:
            print ("WARNING : Truncating", str(len(result))+"-entry LUT to have", aExpectedSize, "entries")
            result = result[0:aExpectedSize]
        elif len(result) != aExpectedSize:
            print ("ERROR parsing file", resolvedPath, ": Expected LUT of size", aExpectedSize, ", but", len(result), "entries were specified (and no padding/truncation requested)")
            sys.exit(1)

        return result


def getFullListOfParameters(aModule):

    def divideByEgLsb(aParam):
        return int(aParam.value() / aModule.egLsb.value())

    def divideByTauLsb(aParam):
        return int(aParam.value() / aModule.tauLsb.value())

    def divideByJetLsb(aParam):
        return int(aParam.value() / aModule.jetLsb.value())


    result = [
      (('mp_common', 'sdfile'),               None,                         ''),
      (('mp_common', 'algoRev'),              None,                         ''),
      (('mp_common', 'leptonSeedThreshold'),  '2_ClusterSeedThreshold.mif', divideByEgLsb(aModule.egSeedThreshold)),
      (('mp_common', 'leptonTowerThreshold'), '3_ClusterThreshold.mif',     divideByEgLsb(aModule.egNeighbourThreshold)),
      (('mp_common', 'pileUpTowerThreshold'), '4_PileUpThreshold.mif',      0x0)
    ]

    result += [
      (('mp_egamma', 'egammaRelaxationThreshold'),  '10_EgRelaxThr.mif',              divideByEgLsb(aModule.egMaxPtHOverE)),
      (('mp_egamma', 'egammaMaxEta'),               'egammaMaxEta.mif',          aModule.egEtaCut.value()),
      (('mp_egamma', 'egammaBypassCuts'),           'BypassEgVeto.mif',               bool(aModule.egBypassEGVetos.value())),
      (('mp_egamma', 'egammaBypassShape'),          'BypassEgShape.mif',              bool(aModule.egBypassShape.value())),
      (('mp_egamma', 'egammaBypassEcalFG'),         'BypassEcalFG.mif',               bool(aModule.egBypassECALFG.value())),
      (('mp_egamma', 'egammaBypassExtendedHOverE'), '_BypassExtHE.mif',               bool(aModule.egBypassExtHOverE)),
      (('mp_egamma', 'egammaHOverECut_iEtaLT15'),   '_RatioCutLt15.mif',              aModule.egHOverEcutBarrel.value()),
      (('mp_egamma', 'egammaHOverECut_iEtaGTEq15'), '_RatioCutGe15.mif',              aModule.egHOverEcutEndcap.value()),
      (('mp_egamma', 'egammaEnergyCalibLUT'),       'C_EgammaCalibration_12to18.mif', parseOfflineLUTfile(aModule.egCalibrationLUTFile.value(), 4096)),
      (('mp_egamma', 'egammaIsoLUT1'),              'D_EgammaIsolation1_13to9.mif',   parseOfflineLUTfile(aModule.egIsoLUTFile.value(), 8192)),
      (('mp_egamma', 'egammaIsoLUT2'),              'D_EgammaIsolation2_13to9.mif',   parseOfflineLUTfile(aModule.egIsoLUTFile2.value(), 8192))
    ]

    result += [
      (('mp_tau', 'tauMaxEta'),         'tauMaxEta.mif',               aModule.isoTauEtaMax.value()),
      (('mp_tau', 'tauEnergyCalibLUT'), 'I_TauCalibration_11to18.mif', parseOfflineLUTfile(aModule.tauCalibrationLUTFile.value(), 2048, 0x0)),
      (('mp_tau', 'tauIsoLUT'),         'H_TauIsolation_12to9.mif',    parseOfflineLUTfile(aModule.tauIsoLUTFile.value(), 4096)),
      (('mp_tau', 'tauTrimmingLUT'),    'P_TauTrimming_13to8.mif',     parseOfflineLUTfile(aModule.tauTrimmingShapeVetoLUTFile.value(), 8192))
    ]

    result += [
      (('mp_jet', 'jetSeedThreshold'),   '1_JetSeedThreshold.mif',      divideByJetLsb(aModule.jetSeedThreshold)),
      (('mp_jet', 'jetMaxEta'),          '6_JetEtaMax.mif',             0x00028),
      (('mp_jet', 'jetBypassPileUpSub'), 'BypassJetPUS.mif',            bool(aModule.jetBypassPUS.value())),
      (('mp_jet', 'jetPUSUsePhiRing'), 'PhiRingPUS.mif',        bool(aModule.jetPUSUsePhiRing.value())),
      (('mp_jet', 'jetEnergyCalibLUT'),  'L_JetCalibration_11to18.mif', parseOfflineLUTfile(aModule.jetCalibrationLUTFile.value(), 2048)),
      (('mp_jet', 'HTMHT_maxJetEta'),    'HTMHT_maxJetEta.mif',         aModule.etSumEtaMax[1]), # assert == etSumEtaMax[3] ?
      (('mp_jet', 'HT_jetThreshold'),    '8_HtThreshold.mif',           int(aModule.etSumEtThreshold[1] / aModule.etSumLsb.value())),
      (('mp_jet', 'MHT_jetThreshold'),   '9_MHtThreshold.mif',          int(aModule.etSumEtThreshold[3] / aModule.etSumLsb.value())),
    ]

    result += [
      (('mp_sums', 'towerCountThreshold'),      'HeavyIonThr.mif',       int(aModule.etSumEtThreshold[4] / aModule.etSumLsb.value()) ),
      (('mp_sums', 'towerCountMaxEta'),         'HeavyIonEta.mif',       aModule.etSumEtaMax[4]),
      (('mp_sums', 'ETMET_maxTowerEta'),        'ETMET_maxTowerEta.mif', aModule.etSumEtaMax[0]), # assert == etSumEtaMax[2] ?
      (('mp_sums', 'ecalET_towerThresholdLUT'), 'X_EcalTHR_11to9.mif',   parseOfflineLUTfile(aModule.etSumEcalSumPUSLUTFile.value(), 2048, aTruncate = True)),
      (('mp_sums', 'ET_towerThresholdLUT'),     'X_ETTHR_11to9.mif',     parseOfflineLUTfile(aModule.etSumEttPUSLUTFile.value(), 2048, aTruncate = True)),
      (('mp_sums', 'MET_towerThresholdLUT'),    'X_METTHR_11to9.mif',    parseOfflineLUTfile(aModule.etSumMetPUSLUTFile.value(), 2048))
    ]

    result += [
      (('demux', 'sdfile'),  None, ''),
      (('demux', 'algoRev'), None, 0xcafe),
      (('demux', 'ET_centralityLowerThresholds'), 'CentralityLowerThrs.mif', [ int(round(loBound / aModule.etSumLsb.value())) for loBound in aModule.etSumCentralityLower.value()]),
      (('demux', 'ET_centralityUpperThresholds'), 'CentralityUpperThrs.mif', [ int(round(upBound / aModule.etSumLsb.value())) for upBound in aModule.etSumCentralityUpper.value()])
      (('demux', 'ET_energyCalibLUT'),            'M_ETMET_11to18.mif',      parseOfflineLUTfile(aModule.etSumEttCalibrationLUTFile.value(), 2048, aTruncate = True)),
      (('demux', 'ecalET_energyCalibLUT'),        'M_ETMETecal_11to18.mif',  parseOfflineLUTfile(aModule.etSumEcalSumCalibrationLUTFile.value(), 2048, aTruncate = True)),
      (('demux', 'METX_energyCalibLUT'),          'M_ETMETX_11to18.mif',     parseOfflineLUTfile(aModule.etSumXCalibrationLUTFile.value(), 2048, aTruncate = True)),
      (('demux', 'METY_energyCalibLUT'),          'M_ETMETY_11to18.mif',     parseOfflineLUTfile(aModule.etSumYCalibrationLUTFile.value(), 2048, aTruncate = True)),
    ]

    result = [(a, b, parseOfflineLUTfile(c.value()) if isinstance(c, cms.FileInPath) else c) for a, b, c in result]

    return result


def getXmlParameterMap(aModule):
    result = {}
    for xmlDetails, mifName, value in getFullListOfParameters(aModule):
        if xmlDetails is not None:
            if xmlDetails[0] in result:
                result[xmlDetails[0]] += [(xmlDetails[1], value)]
            else:
                result[xmlDetails[0]] = [(xmlDetails[1], value)]

    return result


def getMifParameterMap(aModule):

    fullList = getFullListOfParameters(aModule)

    return {mifFileName : value for (_, mifFileName, value) in fullList if mifFileName is not None}


# Stolen from https://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def createMIF(aFilePath, aValue):
    print("Writing MIF file:", aFilePath)
    with open(aFilePath, 'w') as f:
        if isinstance(aValue, bool):
            aValue = (1 if aValue else 0)

        if isinstance(aValue, int):
            f.write( hex(aValue) )
        elif isinstance(aValue, list):
            f.write("\n".join([hex(x) for x in aValue]))
        else:
            raise RuntimeError("Do not know how to deal with parameter of type " + str(type(aValue)))


def createXML(parameters, contextId, outputFilePath):
    topNode = ET.Element('algo', id='calol2')
    contextNode = ET.SubElement(topNode, 'context', id=contextId)
    for paramId, value in parameters:
        if isinstance(value, bool):
            ET.SubElement(contextNode, 'param', id=paramId, type='bool').text = str(value).lower()
        elif isinstance(value, int):
            ET.SubElement(contextNode, 'param', id=paramId, type='uint').text = "0x{0:05X}".format(value)
        elif isinstance(value, str):
            ET.SubElement(contextNode, 'param', id=paramId, type='string').text = value
        elif isinstance(value, list):
            ET.SubElement(contextNode, 'param', id=paramId, type='vector:uint').text  = "\n      " + ",\n      ".join(["0x{0:05X}".format(x) for x in value]) + "\n    "
        else:
            raise RuntimeError("Do not know how to deal with parameter '" + paramId + "' of type " + str(type(value)))
    indent(topNode)

    print("Writing XML file:", outputFilePath)
    with open(outputFilePath, 'w') as f:
        f.write(ET.tostring(topNode))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('params_cfi', help='Name of CMSSW cfi python file specifying the values for the calo parameters')
    parser.add_argument('output_dir', help='Directory for MIF/XML output files')

    outputFormatGroup = parser.add_mutually_exclusive_group(required=True)
    outputFormatGroup.add_argument('--mif', action='store_true')
    outputFormatGroup.add_argument('--xml', action='store_true')

    args = parser.parse_args()

    moduleName = 'L1Trigger.L1TCalorimeter.' + args.params_cfi
    print("Importing calo params from module:", moduleName)
    caloParams = import_module(moduleName).caloStage2Params

    print(caloParams.egCalibrationLUTFile.value())
    print(caloParams.egIsoLUTFile.value())
    print(caloParams.egIsoLUTFile2.value())
    os.mkdir(args.output_dir)

    if args.mif:
        for fileName, value in six.iteritems(getMifParameterMap(caloParams)):
            createMIF(args.output_dir + '/' + fileName, value) 
    else:
        for fileTag, paramList in six.iteritems(getXmlParameterMap(caloParams)):
            createXML(paramList, 'MainProcessor' if fileTag.startswith('mp') else 'Demux', args.output_dir + '/algo_' + fileTag + '.xml')

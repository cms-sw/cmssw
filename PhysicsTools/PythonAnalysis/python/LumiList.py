#!/usr/bin/env python

"""
Handle lists of lumi sections. Constuct in several different formats and filter
(mask) a secondary list of lumis.
This class can also handle ranges of events as the structure is identical
or could be subclassed renaming a function or two.
"""

# An earlier version of this code in in CRAB, but since it's not CRAB specific
# and can be used in config files in general and is now needed for FWLite, it's
# being moved into CMSSW.

__revision__ = "$Id: LumiList.py,v 1.3 2010/06/08 16:08:25 ewv Exp $"
__version__ = "$Revision: 1.3 $"

import json

class LumiList(object):
    """
    Deal with lists of lumis in several different forms:
    Compact list:
        {
        '1': [[1, 33], [35, 35], [37, 47], [49, 75], [77, 130], [133, 136]],
        '2':[[1,45],[50,80]]
        }
        where the first key is the run number, subsequent pairs are
        ranges of lumis within that run that are desired
    Runs and lumis:
        {
        '1': [1,2,3,4,6,7,8,9,10],
        '2': [1,4,5,20]
        }
        where the first key is the run number and the list is a list of
        individual lumi sections
    Run  lumi pairs:
        [[1,1], [1,2],[1,4], [2,1], [2,5], [1,10]]
        where each pair in the list is an individual run&lumi
    CMSSW representation:
        '1:1-1:33,1:35,1:37-1:47,2:1-2:45,2:50-2:80'
        The string used by CMSSW in lumisToProcess or lumisToSkip
        is a subset of the compactList example above
    """


    def __init__(self, filename = None, lumis = None, runsAndLumis = None, runs = None):
        """
        Constructor takes filename (JSON), a list of run/lumi pairs,
        or a dict with run #'s as the keys and a list of lumis as the values, or just a list of runs
        """
        self.compactList = {}
        if filename:
            self.filename = filename
            jsonFile = open(self.filename,'r')
            self.compactList = json.load(jsonFile)
        elif lumis:
            runsAndLumis = {}
            for (run, lumi) in lumis:
                run = str(run)
                if not runsAndLumis.has_key(run):
                    runsAndLumis[run] = []
                runsAndLumis[run].append(lumi)

        if runsAndLumis:
            for run in runsAndLumis.keys():
                runString = str(run)
                lastLumi = -1000
                self.compactList[runString] = []
                lumiList = runsAndLumis[run]
                for lumi in sorted(lumiList):
                    if lumi == lastLumi:
                        pass # Skip duplicates
                    elif lumi != lastLumi + 1: # Break in lumi sequence
                        self.compactList[runString].append([lumi, lumi])
                    else:
                        nRange =  len(self.compactList[runString])
                        self.compactList[runString][nRange-1][1] = lumi
                    lastLumi = lumi
        if runs:
            for run in runs:
                runString = str(run)
                self.compactList[runString] = [[1,0xFFFFFFF]]

    def filterLumis(self, lumiList):
        """
        Return a list of lumis that are in compactList.
        lumilist is of the simple form
        [(run1,lumi1),(run1,lumi2),(run2,lumi1)]
        """
        filteredList = []
        for (run, lumi) in lumiList:
            runsInLumi = self.compactList.get(str(run), [[0,-1]])
            for (first, last) in runsInLumi:
                if lumi >= first and lumi <= last:
                    filteredList.append((run, lumi))
                    break
        return filteredList


    def getCompactList(self):
        """
        Return the compact list representation
        """
        return self.compactList


    def getLumis(self):
        """
        Return the list of pairs representation
        """
        theList = []
        runs = self.compactList.keys()
        runs.sort(key=int)
        for run in runs:
            lumis = self.compactList[run]
            for lumiPair in sorted(lumis):
                for lumi in range(lumiPair[0], lumiPair[1]+1):
                    theList.append((int(run), lumi))

        return theList


    def getCMSSWString(self):
        """
        Turn compactList into a list of the format
        R1:L1,R2:L2-R2:L3 which is acceptable to CMSSW LumiBlockRange variable
        """

        parts = []
        runs = self.compactList.keys()
        runs.sort(key=int)
        for run in runs:
            lumis = self.compactList[run]
            for lumiPair in sorted(lumis):
                if lumiPair[0] == lumiPair[1]:
                    parts.append("%s:%s" % (run, lumiPair[0]))
                else:
                    parts.append("%s:%s-%s:%s" %
                                 (run, lumiPair[0], run, lumiPair[1]))

        output = ','.join(parts)
        return str(output)



# Unit test code
'''
import unittest
from WMCore.DataStructs.LumiList import LumiList


class LumiListTest(unittest.TestCase):
    """
    _LumiListTest_

    """

    def testRead(self):
        """
        Test reading from JSON
        """
        exString = "1:1-1:33,1:35,1:37-1:47,2:49-2:75,2:77-2:130,2:133-2:136"
        exDict   = {'1': [[1, 33], [35, 35], [37, 47]],
                    '2': [[49, 75], [77, 130], [133, 136]]}

        jsonList = LumiList(filename = 'lumiTest.json')
        lumiString = jsonList.getCMSSWString()
        lumiList = jsonList.getCompactList()

        self.assertTrue(lumiString == exString)
        self.assertTrue(lumiList   == exDict)

    def testList(self):
        """
        Test constucting from list of pairs
        """

        listLs1 = range(1, 34) + [35] + range(37, 48)
        listLs2 = range(49, 76) + range(77, 131) + range(133, 137)
        lumis = zip([1]*100, listLs1) + zip([2]*100, listLs2)

        jsonLister = LumiList(filename = 'lumiTest.json')
        jsonString = jsonLister.getCMSSWString()
        jsonList = jsonLister.getCompactList()

        pairLister = LumiList(lumis = lumis)
        pairString = pairLister.getCMSSWString()
        pairList = pairLister.getCompactList()

        self.assertTrue(jsonString == pairString)
        self.assertTrue(jsonList   == pairList)


    def testRuns(self):
        """
        Test constucting from run and list of lumis
        """
        runsAndLumis = {
            1: range(1, 34) + [35] + range(37, 48),
            2: range(49, 76) + range(77, 131) + range(133, 137)
        }
        runsAndLumis2 = {
            '1': range(1, 34) + [35] + range(37, 48),
            '2': range(49, 76) + range(77, 131) + range(133, 137)
        }

        jsonLister = LumiList(filename = 'lumiTest.json')
        jsonString = jsonLister.getCMSSWString()
        jsonList   = jsonLister.getCompactList()

        runLister = LumiList(runsAndLumis = runsAndLumis)
        runString = runLister.getCMSSWString()
        runList   = runLister.getCompactList()

        runLister2 = LumiList(runsAndLumis = runsAndLumis2)
        runList2 = runLister2.getCompactList()

        self.assertTrue(jsonString == runString)
        self.assertTrue(jsonList   == runList)
        self.assertTrue(runList2   == runList)


    def testFilter(self):
        """
        Test filtering of a list of lumis
        """
        runsAndLumis = {
            1: range(1, 34) + [35] + range(37, 48),
            2: range(49, 76) + range(77, 131) + range(133, 137)
        }

        completeList = zip([1]*150, range(1, 150)) + \
                       zip([2]*150, range(1, 150)) + \
                       zip([3]*150, range(1, 150))

        smallList    = zip([1]*50,  range(1, 10)) + zip([2]*50, range(50, 70))
        overlapList  = zip([1]*150, range(30, 40)) + \
                       zip([2]*150, range(60, 80))
        overlapRes   = zip([1]*9,   range(30, 34)) + [(1, 35)] + \
                       zip([1]*9,   range(37, 40)) + \
                       zip([2]*30,  range(60, 76)) + \
                       zip([2]*9,   range(77, 80))

        runLister = LumiList(runsAndLumis = runsAndLumis)

        # Test a list to be filtered which is a superset of constructed list
        filterComplete = runLister.filterLumis(completeList)
        # Test a list to be filtered which is a subset of constructed list
        filterSmall    = runLister.filterLumis(smallList)
        # Test a list to be filtered which is neither
        filterOverlap  = runLister.filterLumis(overlapList)

        self.assertTrue(filterComplete == runLister.getLumis())
        self.assertTrue(filterSmall    == smallList)
        self.assertTrue(filterOverlap  == overlapRes)

    def testDuplicates(self):
        """
        Test a list with lots of duplicates
        """
        result = zip([1]*100, range(1, 34) + range(37, 48))
        lumis  = zip([1]*100, range(1, 34) + range(37, 48) + range(5, 25))

        lister = LumiList(lumis = lumis)
        self.assertTrue(lister.getLumis() == result)

    def testNull(self):
        """
        Test a null list
        """

        runLister = LumiList(lumis = None)

        self.assertTrue(runLister.getCMSSWString() == '')
        self.assertTrue(runLister.getLumis() == [])
        self.assertTrue(runLister.getCompactList() == {})



if __name__ == '__main__':
    unittest.main()
'''

# Test JSON file

#{"1": [[1, 33], [35, 35], [37, 47]], "2": [[49, 75], [77, 130], [133, 136]]}

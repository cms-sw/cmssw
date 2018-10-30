#!/usr/bin/env python

"""
Handle lists of lumi sections. Constuct in several different formats and filter
(mask) a secondary list of lumis.
This class can also handle ranges of events as the structure is identical
or could be subclassed renaming a function or two.

This code began life in COMP/CRAB/python/LumiList.py
"""


import copy
import json
import re
import urllib2

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
        individual lumi sections. This form also takes a list of these objects
        which can be much faster than LumiList += LumiList
    Run  lumi pairs:
        [[1,1], [1,2],[1,4], [2,1], [2,5], [1,10]]
        where each pair in the list is an individual run&lumi
    CMSSW representation:
        '1:1-1:33,1:35,1:37-1:47,2:1-2:45,2:50-2:80'
        The string used by CMSSW in lumisToProcess or lumisToSkip
        is a subset of the compactList example above
    """


    def __init__(self, filename = None, lumis = None, runsAndLumis = None, runs = None, compactList = None, url = None):
        """
        Constructor takes filename (JSON), a list of run/lumi pairs,
        or a dict with run #'s as the keys and a list of lumis as the values, or just a list of runs
        """
        self.compactList = {}
        self.duplicates = {}
        if filename:
            self.filename = filename
            jsonFile = open(self.filename,'r')
            self.compactList = json.load(jsonFile)
        elif url:
            self.url = url
            jsonFile = urllib2.urlopen(url)
            self.compactList = json.load(jsonFile)
        elif lumis:
            runsAndLumis = {}
            for (run, lumi) in lumis:
                run = str(run)
                if run not in runsAndLumis:
                    runsAndLumis[run] = []
                runsAndLumis[run].append(lumi)

        if isinstance(runsAndLumis, list):
            queued = {}
            for runLumiList in runsAndLumis:
                for run, lumis in runLumiList.items():
                    queued.setdefault(run, []).extend(lumis)
            runsAndLumis = queued

        if runsAndLumis:
            for run in runsAndLumis.keys():
                runString = str(run)
                lastLumi = -1000
                lumiList = runsAndLumis[run]
                if lumiList:
                    self.compactList[runString] = []
                    self.duplicates[runString] = []
                    for lumi in sorted(lumiList):
                        if lumi == lastLumi:
                            self.duplicates[runString].append(lumi)
                        elif lumi != lastLumi + 1: # Break in lumi sequence
                            self.compactList[runString].append([lumi, lumi])
                        else:
                            nRange =  len(self.compactList[runString])
                            self.compactList[runString][nRange-1][1] = lumi
                        lastLumi = lumi
        if runs:
            for run in runs:
                runString = str(run)
                self.compactList[runString] = [[1, 0xFFFFFFF]]

        if compactList:
            for run in compactList.keys():
                runString = str(run)
                if compactList[run]:
                    self.compactList[runString] = compactList[run]

        # Compact each run and make it unique

        for run in self.compactList.keys():
            newLumis = []
            for lumi in sorted(self.compactList[run]):
                # If the next lumi starts inside or just after the last just change the endpoint of the first
                if newLumis and newLumis[-1][0] <= lumi[0] <= newLumis[-1][1] + 1:
                    newLumis[-1][1] = max(newLumis[-1][1], lumi[1])
                else:
                    newLumis.append(lumi)
            self.compactList[run] = newLumis

    def __sub__(self, other): # Things from self not in other
        result = {}
        for run in sorted(self.compactList.keys()):
            alumis = sorted(self.compactList[run])
            blumis = sorted(other.compactList.get(run, []))
            alist = []                    # verified part
            for alumi in alumis:
                tmplist = [alumi[0], alumi[1]] # may be part
                for blumi in blumis:
                    if blumi[0] <= tmplist[0] and blumi[1] >= tmplist[1]: # blumi has all of alumi
                        tmplist = []
                        break # blumi is has all of alumi
                    if blumi[0] > tmplist[0] and blumi[1] < tmplist[1]: # blumi is part of alumi
                        alist.append([tmplist[0], blumi[0]-1])
                        tmplist = [blumi[1]+1, tmplist[1]]
                    elif blumi[0] <= tmplist[0] and blumi[1] < tmplist[1] and blumi[1]>=tmplist[0]: # overlaps start
                        tmplist = [blumi[1]+1, tmplist[1]]
                    elif blumi[0] > tmplist[0] and blumi[1] >= tmplist[1] and blumi[0]<=tmplist[1]: # overlaps end
                        alist.append([tmplist[0], blumi[0]-1])
                        tmplist = []
                        break
                if tmplist:
                    alist.append(tmplist)
            result[run] = alist

        return LumiList(compactList = result)


    def __and__(self, other): # Things in both
        result = {}
        aruns = set(self.compactList.keys())
        bruns = set(other.compactList.keys())
        for run in aruns & bruns:
            lumiList = []                    # List for this run
            unique = []                    # List for this run
            for alumi in self.compactList[run]:
                for blumi in other.compactList[run]:
                    if blumi[0] <= alumi[0] and blumi[1] >= alumi[1]: # blumi has all of alumi
                        lumiList.append(alumi)
                    if blumi[0] > alumi[0] and blumi[1] < alumi[1]: # blumi is part of alumi
                        lumiList.append(blumi)
                    elif blumi[0] <= alumi[0] and blumi[1] < alumi[1] and blumi[1] >= alumi[0]: # overlaps start
                        lumiList.append([alumi[0], blumi[1]])
                    elif blumi[0] > alumi[0] and blumi[1] >= alumi[1] and blumi[0] <= alumi[1]: # overlaps end
                        lumiList.append([blumi[0], alumi[1]])


            if lumiList:
                unique = [lumiList[0]]
            for pair in lumiList[1:]:
                if pair[0] == unique[-1][1]+1:
                    unique[-1][1] = copy.deepcopy(pair[1])
                else:
                    unique.append(copy.deepcopy(pair))

            result[run] = unique
        return LumiList(compactList = result)


    def __or__(self, other):
        result = {}
        aruns = self.compactList.keys()
        bruns = other.compactList.keys()
        runs = set(aruns + bruns)
        for run in runs:
            overlap = sorted(self.compactList.get(run, []) + other.compactList.get(run, []))
            unique = [overlap[0]]
            for pair in overlap[1:]:
                if pair[0] >= unique[-1][0] and pair[0] <= unique[-1][1]+1 and pair[1] > unique[-1][1]:
                    unique[-1][1] = copy.deepcopy(pair[1])
                elif pair[0] > unique[-1][1]:
                    unique.append(copy.deepcopy(pair))
            result[run] = unique
        return LumiList(compactList = result)


    def __add__(self, other):
        # + is the same as |
        return self.__or__(other)

    def __len__(self):
        '''Returns number of runs in list'''
        return len(self.compactList)

    def filterLumis(self, lumiList):
        """
        Return a list of lumis that are in compactList.
        lumilist is of the simple form
        [(run1,lumi1),(run1,lumi2),(run2,lumi1)]
        """
        filteredList = []
        for (run, lumi) in lumiList:
            runsInLumi = self.compactList.get(str(run), [[0, -1]])
            for (first, last) in runsInLumi:
                if lumi >= first and lumi <= last:
                    filteredList.append((run, lumi))
                    break
        return filteredList


    def __str__ (self):
        doubleBracketRE = re.compile (r']],')
        return doubleBracketRE.sub (']],\n',
                                    json.dumps (self.compactList,
                                                sort_keys=True))

    def getCompactList(self):
        """
        Return the compact list representation
        """
        return self.compactList


    def getDuplicates(self):
        """
        Return the list of duplicates found during construction as a LumiList
        """
        return LumiList(runsAndLumis = self.duplicates)


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


    def getRuns(self):
        '''
        return the sorted list of runs contained
        '''
        return sorted (self.compactList.keys())


    def _getLumiParts(self):
        """
        Turn compactList into a list of the format
        [ 'R1:L1', 'R2:L2-R2:L3' ] which is used by getCMSSWString and getVLuminosityBlockRange
        """

        parts = []
        runs = self.compactList.keys()
        runs.sort(key=int)
        for run in runs:
            lumis = self.compactList[run]
            for lumiPair in sorted(lumis):
                if lumiPair[0] == lumiPair[1]:
                    parts.append(str("%s:%s" % (run, lumiPair[0])))
                else:
                    parts.append(str("%s:%s-%s:%s" %
                                 (run, lumiPair[0], run, lumiPair[1])))
        return parts


    def getCMSSWString(self):
        """
        Turn compactList into a list of the format
        R1:L1,R2:L2-R2:L3 which is acceptable to CMSSW LumiBlockRange variable
        """

        parts = self._getLumiParts()
        output = ','.join(parts)
        return str(output)


    def getVLuminosityBlockRange(self, tracked = False):
        """
        Turn compactList into an (optionally tracked) VLuminosityBlockRange
        """

        import FWCore.ParameterSet.Config as cms
        parts = self._getLumiParts()
        if tracked:
            return cms.VLuminosityBlockRange(parts)
        else:
            return cms.untracked.VLuminosityBlockRange(parts)


    def writeJSON(self, fileName):
        """
        Write out a JSON file representation of the object
        """
        jsonFile = open(fileName,'w')
        jsonFile.write("%s\n" % self)
        jsonFile.close()


    def removeRuns (self, runList):
        '''
        removes runs from runList from collection
        '''
        for run in runList:
            run = str(run)
            if run in self.compactList:
                del self.compactList[run]

        return


    def selectRuns (self, runList):
        '''
        Selects only runs from runList in collection
        '''
        runsToDelete = []
        for run in self.compactList.keys():
            if int(run) not in runList and run not in runList:
                runsToDelete.append(run)

        for run in runsToDelete:
            del self.compactList[run]

        return

    def contains (self, run, lumiSection = None):
        '''
        returns true if the run, lumi section passed in is contained
        in this lumiList.  Input can be either:
        - a single tuple of (run, lumi),
        - separate run and lumi numbers
        - a single run number (returns true if any lumi sections exist)
        '''
        if lumiSection is None:
            # if this is an integer or a string, see if the run exists
            if isinstance (run, int) or isinstance (run, str):
                return str(run) in self.compactList
            # if we're here, then run better be a tuple or list
            try:
                lumiSection = run[1]
                run         = run[0]
            except:
                raise RuntimeError("Improper format for run '%s'" % run)
        lumiRangeList = self.compactList.get( str(run) )
        if not lumiRangeList:
            # the run isn't there, so no need to look any further
            return False
        for lumiRange in lumiRangeList:
            # we want to make this as found if either the lumiSection
            # is inside the range OR if the lumi section is greater
            # than or equal to the lower bound of the lumi range and
            # the upper bound is 0 (which means extends to the end of
            # the run)
            if lumiRange[0] <= lumiSection and \
               (0 == lumiRange[1] or lumiSection <= lumiRange[1]):
                # got it
                return True
        return False


    def __contains__ (self, runTuple):
        return self.contains (runTuple)



'''
# Unit test code
import unittest
import FWCore.ParameterSet.Config as cms

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
        exVLBR   = cms.VLuminosityBlockRange('1:1-1:33', '1:35', '1:37-1:47', '2:49-2:75', '2:77-2:130', '2:133-2:136')

        jsonList = LumiList(filename = 'lumiTest.json')
        lumiString = jsonList.getCMSSWString()
        lumiList = jsonList.getCompactList()
        lumiVLBR = jsonList.getVLuminosityBlockRange(True)

        self.assertTrue(lumiString == exString)
        self.assertTrue(lumiList   == exDict)
        self.assertTrue(lumiVLBR   == exVLBR)

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
        blank = {
            '1': [],
            '2': []
        }

        jsonLister = LumiList(filename = 'lumiTest.json')
        jsonString = jsonLister.getCMSSWString()
        jsonList   = jsonLister.getCompactList()

        runLister = LumiList(runsAndLumis = runsAndLumis)
        runString = runLister.getCMSSWString()
        runList   = runLister.getCompactList()

        runLister2 = LumiList(runsAndLumis = runsAndLumis2)
        runList2 = runLister2.getCompactList()

        runLister3 = LumiList(runsAndLumis = blank)


        self.assertTrue(jsonString == runString)
        self.assertTrue(jsonList   == runList)
        self.assertTrue(runList2   == runList)
        self.assertTrue(len(runLister3) == 0)

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

    def testSubtract(self):
        """
        a-b for lots of cases
        """

        alumis = {'1' : range(2,20) + range(31,39) + range(45,49),
                  '2' : range(6,20) + range (30,40),
                  '3' : range(10,20) + range (30,40) + range(50,60),
                 }
        blumis = {'1' : range(1,6) + range(12,13) + range(16,30) + range(40,50) + range(33,36),
                  '2' : range(10,35),
                  '3' : range(10,15) + range(35,40) + range(45,51) + range(59,70),
                 }
        clumis = {'1' : range(1,6) + range(12,13) + range(16,30) + range(40,50) + range(33,36),
                  '2' : range(10,35),
                 }
        result = {'1' : range(6,12) + range(13,16) + range(31,33) + range(36,39),
                  '2' : range(6,10) + range(35,40),
                  '3' : range(15,20) + range(30,35) + range(51,59),
                 }
        result2 = {'1' : range(6,12) + range(13,16) + range(31,33) + range(36,39),
                   '2' : range(6,10) + range(35,40),
                   '3' : range(10,20) + range (30,40) + range(50,60),
                 }
        a = LumiList(runsAndLumis = alumis)
        b = LumiList(runsAndLumis = blumis)
        c = LumiList(runsAndLumis = clumis)
        r = LumiList(runsAndLumis = result)
        r2 = LumiList(runsAndLumis = result2)

        self.assertTrue((a-b).getCMSSWString() == r.getCMSSWString())
        self.assertTrue((a-b).getCMSSWString() != (b-a).getCMSSWString())
        # Test where c is missing runs from a
        self.assertTrue((a-c).getCMSSWString() == r2.getCMSSWString())
        self.assertTrue((a-c).getCMSSWString() != (c-a).getCMSSWString())
        # Test empty lists
        self.assertTrue(str(a-a) == '{}')
        self.assertTrue(len(a-a) == 0)

    def testOr(self):
        """
        a|b for lots of cases
        """

        alumis = {'1' : range(2,20) + range(31,39) + range(45,49),
                  '2' : range(6,20) + range (30,40),
                  '3' : range(10,20) + range (30,40) + range(50,60),
                 }
        blumis = {'1' : range(1,6) + range(12,13) + range(16,30) + range(40,50) + range(39,80),
                  '2' : range(10,35),
                  '3' : range(10,15) + range(35,40) + range(45,51) + range(59,70),
                 }
        clumis = {'1' : range(1,6) + range(12,13) + range(16,30) + range(40,50) + range(39,80),
                  '2' : range(10,35),
                 }
        result = {'1' : range(2,20) + range(31,39) + range(45,49) + range(1,6) + range(12,13) + range(16,30) + range(40,50) + range(39,80),
                  '2' : range(6,20) + range (30,40) + range(10,35),
                  '3' : range(10,20) + range (30,40) + range(50,60) + range(10,15) + range(35,40) + range(45,51) + range(59,70),
                 }
        a = LumiList(runsAndLumis = alumis)
        b = LumiList(runsAndLumis = blumis)
        c = LumiList(runsAndLumis = blumis)
        r = LumiList(runsAndLumis = result)
        self.assertTrue((a|b).getCMSSWString() == r.getCMSSWString())
        self.assertTrue((a|b).getCMSSWString() == (b|a).getCMSSWString())
        self.assertTrue((a|b).getCMSSWString() == (a+b).getCMSSWString())

        # Test list constuction (faster)

        multiple = [alumis, blumis, clumis]
        easy = LumiList(runsAndLumis = multiple)
        hard = a + b
        hard += c
        self.assertTrue(hard.getCMSSWString() == easy.getCMSSWString())

    def testAnd(self):
        """
        a&b for lots of cases
        """

        alumis = {'1' : range(2,20) + range(31,39) + range(45,49),
                  '2' : range(6,20) + range (30,40),
                  '3' : range(10,20) + range (30,40) + range(50,60),
                  '4' : range(1,100),
                 }
        blumis = {'1' : range(1,6) + range(12,13) + range(16,25) + range(25,40) + range(40,50) + range(33,36),
                  '2' : range(10,35),
                  '3' : range(10,15) + range(35,40) + range(45,51) + range(59,70),
                  '5' : range(1,100),
                 }
        result = {'1' : range(2,6) + range(12,13) + range(16,20) + range(31,39) + range(45,49),
                  '2' : range(10,20) + range(30,35),
                  '3' : range(10,15) + range(35,40) + range(50,51)+ range(59,60),
                 }
        a = LumiList(runsAndLumis = alumis)
        b = LumiList(runsAndLumis = blumis)
        r = LumiList(runsAndLumis = result)
        self.assertTrue((a&b).getCMSSWString() == r.getCMSSWString())
        self.assertTrue((a&b).getCMSSWString() == (b&a).getCMSSWString())
        self.assertTrue((a|b).getCMSSWString() != r.getCMSSWString())

    def testRemoveSelect(self):
        """
        a-b for lots of cases
        """

        alumis = {'1' : range(2,20) + range(31,39) + range(45,49),
                  '2' : range(6,20) + range (30,40),
                  '3' : range(10,20) + range (30,40) + range(50,60),
                  '4' : range(10,20) + range (30,80),
                 }

        result = {'2' : range(6,20) + range (30,40),
                  '4' : range(10,20) + range (30,80),
                 }

        rem = LumiList(runsAndLumis = alumis)
        sel = LumiList(runsAndLumis = alumis)
        res = LumiList(runsAndLumis = result)

        rem.removeRuns([1,3])
        sel.selectRuns([2,4])

        self.assertTrue(rem.getCMSSWString() == res.getCMSSWString())
        self.assertTrue(sel.getCMSSWString() == res.getCMSSWString())
        self.assertTrue(sel.getCMSSWString() == rem.getCMSSWString())

    def testURL(self):
        URL = 'https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions12/8TeV/Reprocessing/Cert_190456-195530_8TeV_08Jun2012ReReco_Collisions12_JSON.txt'
        ll = LumiList(url=URL)
        self.assertTrue(len(ll) > 0)


    def testWrite(self):
        alumis = {'1' : range(2,20) + range(31,39) + range(45,49),
                  '2' : range(6,20) + range (30,40),
                  '3' : range(10,20) + range (30,40) + range(50,60),
                  '4' : range(1,100),
                 }
        a = LumiList(runsAndLumis = alumis)
        a.writeJSON('newFile.json')


if __name__ == '__main__':
    jsonFile = open('lumiTest.json','w')
    jsonFile.write('{"1": [[1, 33], [35, 35], [37, 47]], "2": [[49, 75], [77, 130], [133, 136]]}')
    jsonFile.close()
    unittest.main()
'''
# Test JSON file

#{"1": [[1, 33], [35, 35], [37, 47]], "2": [[49, 75], [77, 130], [133, 136]]}

if __name__ == '__main__':
    #############################################
    ## Load and save command line history when ##
    ## running interactively.                  ##
    #############################################
    import os, readline
    import atexit
    historyPath = os.path.expanduser("~/.pyhistory")


    def save_history(historyPath=historyPath):
        import readline
        readline.write_history_file(historyPath)
        if os.path.exists(historyPath):
            readline.read_history_file(historyPath)


    atexit.register(save_history)
    readline.parse_and_bind("set show-all-if-ambiguous on")
    readline.parse_and_bind("tab: complete")
    if os.path.exists (historyPath) :
        readline.read_history_file(historyPath)
        readline.set_history_length(-1)



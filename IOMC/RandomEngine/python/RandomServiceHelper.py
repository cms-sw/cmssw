#!/usr/bin/env python


from FWCore.ParameterSet.Config import Service
import FWCore.ParameterSet.Types as CfgTypes


class RandomNumberServiceHelper(object):
    """
    _RandomNumberServiceHelper_

    Helper class to hold and handle the Random number generator service.

    Provide both user level and WM APIs.

    Revision: "$Id: RandomServiceHelper.py,v 1.4 2008/06/10 19:49:49 ewv Exp $"
    Version   "$Revision: 1.4 $"
    Author:   Dave Evans
    Modified: Eric Vaandering
    """

    def __init__(self,randService):
        self._randService = randService
        self._lockedSeeds = []


    def __containsSeed(self,psetInstance):
        """
        _keeper_

        True/False if the psetInstance has seeds in it

        """
        if psetInstance == None:
            return False
        if not isinstance(psetInstance,CfgTypes.PSet):
            return False
        seedList = getattr(psetInstance, "initialSeedSet", None)
        if seedList != None:
            return True
        seedVal = getattr(psetInstance, "initialSeed", None)
        if seedVal != None:
            return True
        return False


    def __psetsWithSeeds(self):
        """
        _psetsWithSeeds_

        *private method*

        return the list of PSet instances with seeds in them

        """
        svcAttrs = [getattr(self._randService, item, None)
                    for item in self._randService.parameters_()
                    if item not in self._lockedSeeds]

        #print svcAttrs

        return filter(self.__containsSeed, svcAttrs)


    def countSeeds(self):
        """
        _countSeeds_

        Count the number of seeds required by this service by
        summing up the initialSeed and initialSeedSet entries
        in all PSets in the service that contain those parameters.

        """
        count = 0

        for itemRef in self.__psetsWithSeeds():
            #  //
            # // PSet has list of seeds
            #//
            seedSet = getattr(itemRef, "initialSeedSet", None)
            if seedSet != None:
                count += len( seedSet.value())
                continue
            #  //
            # // PSet has single seed
            #//
            seedVal =  getattr(itemRef, "initialSeed", None)
            if seedVal != None:
                count += 1

            #  //
            # // PSet has no recognisable seed, therfore do nothing
            #//  with it
        return count


    def setNamedSeed(self, psetName, *seeds):
        """
        _setNamedSeed_

        If a specific set of seeds is needed for a PSet in this
        service, they can be set by name using this method.

        - *psetName* : Name of the pset containing the seeds

        - *seeds*    : list of seeds to be added, should be a single seed
        for initialSeed values.

        """
        pset = getattr(self._randService, psetName, None)
        if pset == None:
            msg = "No PSet named %s belongs to this instance of the" % (
                psetName,)
            msg += "Random Seed Service"
            raise RuntimeError, msg

        seedVal = getattr(pset, "initialSeed", None)
        if seedVal != None:
            pset.initialSeed = CfgTypes.untracked(
                CfgTypes.uint32(seeds[0])
                )

            return
        seedSet = getattr(pset, "initialSeedSet", None)
        if seedSet != None:
            #  //
            # // Do we want to check the number of seeds??
            #//
            #if len(seeds) != len( seedSet.value()): pass
            pset.initialSeedSet = CfgTypes.untracked(
                CfgTypes.vuint32(*seeds))
            return
        #  //
        # // No seeds for that PSet
        #//  Error throw?
        return


    def getNamedSeed(self, psetName):
        """
        _getNamedSeed_

        This method returns the seeds in a PSet in this service. Returned

        - *psetName* : Name of the pset containing the seeds

        """
        pset = getattr(self._randService, psetName, None)
        if pset == None:
            msg = "No PSet named %s belongs to this instance of the" % (
                psetName,)
            msg += "Random Seed Service"
            raise RuntimeError, msg

        seedVal = getattr(pset, "initialSeed", None)
        if seedVal != None:
            return [pset.initialSeed.value()]

        seedSet = getattr(pset, "initialSeedSet", None)
        if seedSet != None:
            return pset.initialSeedSet


    def insertSeeds(self, *seeds):
        """
        _insertSeeds_

        Given some list of specific seeds, insert them into the
        service.

        Length of seed list is required to be same as the seed count for
        the service.

        Usage: WM Tools.

        """
        seeds = list(seeds)
        if len(seeds) < self.countSeeds():
            msg = "Not enough seeds provided\n"
            msg += "Service requires %s seeds, only %s provided\n"
            msg += "to RandomeService.insertSeeds method\n"
            raise RuntimeError, msg

        for item in self.__psetsWithSeeds():
            seedSet = getattr(item, "initialSeedSet", None)
            if seedSet != None:
                numSeeds = len(seedSet.value())
                useSeeds = seeds[:numSeeds]
                seeds = seeds[numSeeds:]
                item.initialSeedSet = CfgTypes.untracked(
                    CfgTypes.vuint32(*useSeeds))
                continue
            useSeed = seeds[0]
            seeds = seeds[1:]
            item.initialSeed = CfgTypes.untracked(
                CfgTypes.uint32(useSeed)
                )
            continue
        return


    def populate(self, *excludePSets):
        """
        _populate_

        generate a bunch of seeds and stick them into this service
        This is the lazy user method.

        Optional args are names of PSets to *NOT* alter seeds.

        Eg:
        populate() will set all seeds
        populate("pset1", "pset2") will set all seeds but not those in
        psets named pset1 and pset2

        """

        import random
        from random import SystemRandom
        _inst = SystemRandom()
        _MAXINT = 900000000

        #  //
        # // count seeds and create the required number of seeds
        #//
        newSeeds = [ _inst.randint(1, _MAXINT)
                     for i in range(self.countSeeds())]


        self._lockedSeeds = list(excludePSets)
        self.insertSeeds(*newSeeds)
        self._lockedSeeds = []
        return


    def resetSeeds(self, value):
        """
        _resetSeeds_

        reset all seeds to given value

        """
        newSeeds = [ value for i in range(self.countSeeds())]
        self.insertSeeds(*newSeeds)
        return



if __name__ == '__main__':
    #  //
    # // Setup a test service and populate it
    #//
    randSvc = Service("RandomNumberGeneratorService")
    randHelper = RandomNumberServiceHelper(randSvc)

    randSvc.i1 =  CfgTypes.untracked(CfgTypes.uint32(1))
    randSvc.t1 = CfgTypes.PSet()
    randSvc.t2 = CfgTypes.PSet()
    randSvc.t3 = CfgTypes.PSet()

    randSvc.t1.initialSeed = CfgTypes.untracked(
        CfgTypes.uint32(123455678)
        )

    randSvc.t2.initialSeedSet = CfgTypes.untracked(
        CfgTypes.vuint32(12345,234567,345677)
        )


    randSvc.t3.initialSeed = CfgTypes.untracked(
        CfgTypes.uint32(987654321)
        )

    print "Inital PSet"
    print randSvc


    #  //
    # // Autofill seeds
    #//
    print "Totally Random PSet"
    randHelper.populate()
    print randSvc


    #  //
    # // Set all seeds with reset method
    #//
    print "All seeds 9999"
    randHelper.resetSeeds(9999)
    print randSvc

    #  //
    # // test setting named seeds
    #//
    print "t1,t3 9998"
    randHelper.setNamedSeed("t1", 9998)
    randHelper.setNamedSeed("t3", 9998, 9998)
    print randSvc

    print "t1 seed(s)",randHelper.getNamedSeed("t1")
    print "t2 seed(s)",randHelper.getNamedSeed("t2")


    #  //
    # // Autofill seeds with exclusion list
    #//
    randHelper.populate("t1", "t3")
    print "t2 randomized"
    print randSvc

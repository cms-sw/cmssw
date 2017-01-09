#!/usr/bin/env python

##########################################################################
# Classes which provide the geometry information.
##

import itertools
import os

from ROOT import TFile, TTree

from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate import geometrydata


class Alignables:
    """ Creates a list of the aligned strucutres. Get the fields out of the
    TrackerTree.root file.
    """

    def __init__(self, config):
        # list of Structure objects, contains structures which were aligned
        self.structures = []
        self.config = config

    def get_name_by_objid(self, objid):
        return geometrydata.data[objid].name

    def get_subdetid(self, objid):
        return geometrydata.data[objid].subdetid

    def get_discriminator(self, objid):
        return geometrydata.data[objid].discriminator

    def get_ndiscriminator(self, objid):
        return geometrydata.data[objid].ndiscriminator

    def create_list(self, MillePedeUser):
        # loop over output TTree
        for line in MillePedeUser:
            # check which structures were aligned
            if (line.ObjId != 1 and 999999 not in map(abs, line.Par)):
                # check if structure is not yet in the list
                if not any(x.name == self.get_name_by_objid(line.ObjId) for x in self.structures):
                    # create new structure object
                    name = self.get_name_by_objid(line.ObjId)
                    subdetid = self.get_subdetid(line.ObjId)
                    discriminator = self.get_discriminator(line.ObjId)
                    ndiscriminator = self.get_ndiscriminator(line.ObjId)
                    # create structure
                    self.structures.append(
                        Structure(name, subdetid, discriminator, ndiscriminator))
                    # add detids which belong to this structure
                    self.structures[-1].detids = self.get_detids(subdetid)

    def create_children_list(self):
        for struct in self.structures:
            # loop over discriminators -> create patterns
            # pattern {"half": 2, "side": 2, "layer": 6, ...}
            ranges = struct.ndiscriminator
            pranges = [range(1, x+1) for x in ranges]
            # loop over all possible combinations of the values of the
            # discriminators
            for number in itertools.product(*pranges):
                # create pattern dict
                pattern = dict(zip(struct.discriminator, number))
                # name out of patten
                name = " ".join("{0} {1}".format(key, value)
                                for (key, value) in pattern.items())
                # get detids of child
                detids = self.get_detids(struct.subdetid, pattern)
                # create child and add it to parent
                child = Structure(name, struct.subdetid, detids=detids)
                struct.children.append(child)

    def get_detids(self, subdetid, pattern={}):
        # list of all detids in the structure
        detids = []
        # open TrackerTree.root file
        treeFile = TFile(os.path.join(self.config.mpspath, "TrackerTree.root"))
        tree = treeFile.Get("TrackerTreeGenerator/TrackerTree/TrackerTree")

        for line in tree:
            # check if line is part of the structure
            if (line.SubdetId == subdetid):

                # to create a child also check the pattern
                if ("half" in pattern):
                    if (line.Half != pattern["half"]):
                        continue

                if ("side" in pattern):
                    if (line.Side != pattern["side"]):
                        continue

                if ("layer" in pattern):
                    if (line.Layer != pattern["layer"]):
                        continue

                if ("rod" in pattern):
                    if (line.Rod != pattern["rod"]):
                        continue

                if ("ring" in pattern):
                    if (line.Ring != pattern["ring"]):
                        continue

                if ("petal" in pattern):
                    if (line.Petal != pattern["petal"]):
                        continue

                if ("blade" in pattern):
                    if (line.Blade != pattern["blade"]):
                        continue

                if ("panel" in pattern):
                    if (line.Panel != pattern["panel"]):
                        continue

                if ("outerinner" in pattern):
                    if (line.OuterInner != pattern["outerinner"]):
                        continue

                if ("module" in pattern):
                    if (line.Module != pattern["module"]):
                        continue

                detids.append(line.RawId)
        return detids


class Structure:
    """ A object represents a physical strucutre
    """

    def __init__(self, name, subdetid, discriminator=[], ndiscriminator=[], detids=[]):
        # name of the structure
        self.name = name
        # fields to identify the DetIds which belong to the structure
        self.subdetid = subdetid
        # fields which allow to discriminate the parts of the structure
        self.discriminator = discriminator
        # number per discriminator
        self.ndiscriminator = ndiscriminator
        # all DetIds which belong to this structure
        self.detids = detids
        # fieldss of all parts of the structure
        self.children = []

    def get_name(self):
        return self.name

    def get_children(self):
        return self.children

    def contains_detid(self, detid):
        if detid in self.detids:
            return True
        return False

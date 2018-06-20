##########################################################################
# Classes which provide the geometry information.
##

import itertools
import os

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch()

import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.geometrydata as mpsv_geometrydata


class Alignables:
    """ Creates a list of the aligned strucutres. Get the fields out of the
    TrackerTree.root file.
    """

    def __init__(self, config):
        # list of Structure objects, contains structures which were aligned
        self.structures = []
        self.config = config

    def get_subdetid(self, objid):
        return mpsv_geometrydata.data[objid].subdetid

    def get_discriminator(self, objid):
        return mpsv_geometrydata.data[objid].discriminator

    def get_ndiscriminator(self, objid):
        subdetid = self.get_subdetid(objid)
        discriminator = self.get_discriminator(objid)
        ndiscriminator = {key: [] for key in discriminator}
        # open TrackerTree.root file
        treeFile = ROOT.TFile(os.path.join(self.config.jobDataPath,
                                           ".TrackerTree.root"))
        tree = treeFile.Get("TrackerTreeGenerator/TrackerTree/TrackerTree")

        for entry in tree:
            # check if entry is part of the structure
            if (entry.SubdetId == subdetid):
                for structure in discriminator:
                    ndiscriminator[structure].append(getattr(entry, structure))
        for structure in discriminator:
            ndiscriminator[structure] = [x for x in ndiscriminator[structure] if x != 0]

        return [len(set(ndiscriminator[structure]))
                for structure in discriminator]

    def create_list(self, MillePedeUser):
        # loop over output TTree
        for entry in MillePedeUser:
            # check which structures were aligned
            if (entry.ObjId != 1 and 999999 not in map(abs, entry.Par)):
                # check if structure is not yet in the list
                if not any(x.name == str(entry.Name) for x in self.structures):
                    # create new structure object
                    name = str(entry.Name)
                    subdetid = self.get_subdetid(entry.ObjId)
                    discriminator = self.get_discriminator(entry.ObjId)
                    ndiscriminator = self.get_ndiscriminator(entry.ObjId)
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
                pattern = dict(zip(map(lambda x: x.lower(), struct.discriminator), number))
                # name out of pattern
                name = " ".join("{0} {1}".format(key.lower(), value)
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
        treeFile = ROOT.TFile(os.path.join(self.config.jobDataPath,
                                           ".TrackerTree.root"))
        tree = treeFile.Get("TrackerTreeGenerator/TrackerTree/TrackerTree")

        for entry in tree:
            # check if entry is part of the structure
            if (entry.SubdetId == subdetid):
                # to create a child also check the pattern
                structure_found = False
                for structure in ("Half", "Side", "Layer", "Rod", "Ring",
                                  "Petal", "Blade", "Panel", "OuterInner",
                                  "Module"):
                    if structure.lower() in pattern:
                        if getattr(entry, structure) != pattern[structure.lower()]:
                            structure_found = True
                            break
                if structure_found: continue

                detids.append(entry.RawId)
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

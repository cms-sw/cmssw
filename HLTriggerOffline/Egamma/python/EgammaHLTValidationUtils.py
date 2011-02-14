#!/usr/bin/env python
import FWCore.ParameterSet.Config as cms
import sys, os

# prefix for printouts
msgPrefix = "[" + os.path.basename(__file__) + "]"


#----------------------------------------------------------------------

def getPathsOfDataSet(process, datasetName):
    """ returns the names of the trigger paths contained in the
        given (primary) dataset """

    return list(getattr(process.datasets, datasetName))

#----------------------------------------------------------------------


def getProcessName(pdgGen, requiredNumberOfGeneratedObjects):
    """ returns a process name (such as 'Zee') which can be
     used in various places (e.g. module names etc.) """

    if pdgGen == 11:

        # electron paths
        if requiredNumberOfGeneratedObjects == 1:
            return "Wenu"
        elif requiredNumberOfGeneratedObjects == 2:
            return "Zee"
        else:
            raise Exception("unsupported case, can't guess type of process")

    elif pdgGen == 22:

        # photon paths
        if requiredNumberOfGeneratedObjects == 1:
            return 'GammaJet'
        elif requiredNumberOfGeneratedObjects == 2:
            return 'DiGamma'
        else:
            raise Exception("unsupported case, can't guess type of process")
    else:
        raise Exception("unsupported case, can't guess type of process")


#----------------------------------------------------------------------

def makeGeneratedParticleAndFiducialVolumeFilter(process, pdgGen, requiredNumberOfGeneratedObjects):
    """
    adds the needed modules to the process object and
    returns a sequence made of the two filters.

    returns the name of the created module

    if process is not None, are added to the process.

    When using this function from a _cff file, one
    has to manually add the modules of the returned
    sequence to globals() (in the calling module, not
    here, globals() live in a different name space here)
    
    """

    # name of the physics process
    procName = getProcessName(pdgGen, requiredNumberOfGeneratedObjects)

    #--------------------
    # create a module producing a collection with the 
    # desired generated particles
    #--------------------

    genPartModuleName = 'genpart' + procName


    genPartModule = cms.EDFilter("PdgIdAndStatusCandViewSelector",
                                 status = cms.vint32(3),
                                 src = cms.InputTag("genParticles"),
                                 pdgId = cms.vint32(pdgGen),
                                 )

    # genPartModule.setLabel(genPartModuleName)
    if process != None:
        setattr(process, genPartModuleName, genPartModule)

    genPartModule.setLabel(genPartModuleName)

    #--------------------
    # create a module requiring the number
    # of generated particles
    #--------------------

    selectorModuleName = "fiducial" + procName

    selectorModule = cms.EDFilter("EtaPtMinCandViewSelector",
                                  src = cms.InputTag(genPartModuleName),
                                  etaMin = cms.double(-2.5),  
                                  etaMax = cms.double(2.5),   
                                  ptMin = cms.double(2.0)
                                  )

    if process != None:
        setattr(process, selectorModuleName, selectorModule)

    # this is needed if we don't have a process to attach this module to
    selectorModule.setLabel(selectorModuleName)

    #--------------------
    # create the sequence
    #--------------------

    return cms.Sequence(
        # getattr(process, genPartModuleName)
        genPartModule 

        *

        # getattr(process, selectorModuleName)
        selectorModule
        )
#----------------------------------------------------------------------

import HLTriggerOffline.Egamma.TriggerTypeDefs_cfi as TriggerTypeDefs_cfi

class EgammaDQMModuleMaker:
    """ a class which can be used to produce an analysis path
        for the EmDQM analyzer """

    #----------------------------------------

    def __init__(self, process, pathName, pdgGen, requiredNumberOfGeneratedObjects, cutCollection = None):
        """
        pathName is the HLT path to be validated.

        pdgGen is the PDG id of the corersponding generated particles
          (11 for electrons, 22 for photons)

        requiredNumberOfGeneratedObjects should be 1 for single triggers,
        and 2 for double triggers (e.g. double photon triggers)

        cutCollection is the name of the collection which should be used
          to define the acceptance region (at reconstruction level ?).
          typical values are 'fiducialZee'. If this is set to None,
          will be determined automatically from pdgGen and requiredNumberOfGeneratedObjects

        """

        self.process = process
        self.pathName = pathName

        self.path = getattr(process,pathName)

        # the process whose products should be analyzed
        self.processName = "HLT"

        #--------------------
        # guess the collection for the fiducial volume cut
        #--------------------

        if cutCollection == None:
            cutCollection = "fiducial" + getProcessName(pdgGen, requiredNumberOfGeneratedObjects)

        #--------------------
        # initialize the analyzer we put together here
        #--------------------
        self.__result = cms.EDAnalyzer("EmDQM",
                                     triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),                            
                                     genEtaAcc = cms.double(2.5),
                                     genEtAcc = cms.double(2.0),
                                     reqNum = cms.uint32(requiredNumberOfGeneratedObjects),
                                     filters = cms.VPSet(), # will be added later
                                     PtMax = cms.untracked.double(100.0),
                                     pdgGen = cms.int32(pdgGen),
                                     cutcollection = cms.InputTag(cutCollection),

                                     # is this a requirement on reconstructed or generated number of objects ?
                                     cutnum = cms.int32(requiredNumberOfGeneratedObjects),


                                       
                                       )

        #--------------------
        # get all modules of this path.
        # dirty hack: assumes that all modules
        #   are concatenated by '+'
        # but easier than to use a node visitor
        # and order the modules ourselves afterwards..

        moduleNames = str(self.path).split('+')

        # now find out which of these are EDFilters
        # and what CMSSW class type they are

        # example:
        #
        # CMSSW type                               module name
        # --------------------------------------------------------------------------------------------------------------------
        # HLTTriggerTypeFilter                     hltTriggerType 
        # HLTLevel1GTSeed                          hltL1sL1SingleEG8 
        # HLTPrescaler                             hltPreEle17SWTighterEleIdIsolL1R 
        # HLTEgammaL1MatchFilterRegional           hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolL1MatchFilterRegional 
        # HLTEgammaEtFilter                        hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolEtFilter 
        # HLTEgammaGenericFilter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolR9ShapeFilter 
        # HLTEgammaGenericFilter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolClusterShapeFilter 
        # HLTEgammaGenericFilter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TIghterEleIdIsolEcalIsolFilter 
        # HLTEgammaGenericFilter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHEFilter 
        # HLTEgammaGenericFilter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHcalIsolFilter 
        # HLTElectronPixelMatchFilter              hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolPixelMatchFilter 
        # HLTElectronOneOEMinusOneOPFilterRegional hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolOneOEMinusOneOPFilter 
        # HLTElectronGenericFilter                 hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDetaFilter 
        # HLTElectronGenericFilter                 hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDphiFilter 
        # HLTElectronGenericFilter                 hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolTrackIsolFilter 
        # HLTBool                                  hltBoolEnd 

        # it looks like in the MC menu, all modules have a name containing 'L1NonIso' and then
        # have a parameter IsoCollections (which is mostly cms.Input("none")...)

        import FWCore.ParameterSet.Modules

        for moduleName in moduleNames:

            # add protection to avoid accessing non-existing modules
            # (not understood why this is needed but happens
            # in some cases...).
            #
            # should also cover the special cases listed afterwards
            # (i.e. the check after this one could be removed
            # at some point)
            if not hasattr(self.process, moduleName):
                continue

            # protection for FastSim HLT menu
            # which seems to reference certain modules in some
            # paths but these modules are not defined when just
            # loading the HLT menu into a process object.
            #
            # this seems not to happen for the fullsim menu for some
            # reason...
            if moduleName in ('simulation',
                              'offlineBeamSpot',
                              'HLTEndSequence'):
                continue

            

            module = getattr(self.process,moduleName)

            if not isinstance(module, FWCore.ParameterSet.Modules.EDFilter):
                continue

            # ignore certain EDFilters
            if module.type_() in ('HLTTriggerTypeFilter',
                                  'HLTPrescaler',
                                  'HLTBool'):
                continue

            # print "XX", module.type_(), moduleName

            #--------------------
            if module.type_() == 'HLTLevel1GTSeed':
                # L1 seed
                self.__result.filters.append(self.makePSetForL1SeedFilter(moduleName))
                continue

            #--------------------
            if module.type_() == 'HLTEgammaL1MatchFilterRegional':
                # L1 seed to supercluster match
                self.__result.filters.append(self.makePSetForL1SeedToSuperClusterMatchFilter(moduleName))
                continue

            #--------------------

            if module.type_() == "HLTEgammaEtFilter":
                # minimum Et requirement
                self.__result.filters.append(self.makePSetForEtFilter(moduleName))
                continue

            #--------------------

            if module.type_() == "HLTElectronOneOEMinusOneOPFilterRegional":
                self.__result.filters.append(self.makePSetForOneOEMinusOneOPFilter(moduleName))
                continue

            #--------------------
            if module.type_() == "HLTElectronPixelMatchFilter":
                self.__result.filters.append(self.makePSetForPixelMatchFilter(moduleName))
                continue

            #--------------------
            # generic filters: the module types
            # aren't enough, we must check on which
            # input collections they filter on
            #--------------------

            if module.type_() == "HLTEgammaGenericFilter":

                pset = self.makePSetForEgammaGenericFilter(module, moduleName)
                if pset != None:
                    self.__result.filters.append(pset)
                    continue

            #--------------------

            if module.type_() == "HLTElectronGenericFilter":

                pset = self.makePSetForElectronGenericFilter(module, moduleName)
                if pset != None:
                    self.__result.filters.append(pset)
                    continue

            #--------------------

##            print >> sys.stderr,msgPrefix,"WARNING: unknown module type", module.type_(), " with name " + moduleName + " in path " + pathName
                                         
    #----------------------------------------
    
    def makePSetForL1SeedFilter(self,moduleName):
        """ generates a PSet to analyze the behaviour of an L1 seed.

            moduleName is the name of the HLT module which filters
            on the L1 seed.
        """

        return cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(TriggerTypeDefs_cfi.TriggerL1NoIsoEG)
        ) 

    #----------------------------------------

    def makePSetForL1SeedToSuperClusterMatchFilter(self,moduleName):
        """ generates a PSet to analyze the behaviour of L1 to supercluster match filter.

            moduleName is the name of the HLT module which requires the match
            between supercluster and L1 seed.
        """

        return cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(TriggerTypeDefs_cfi.TriggerCluster)
        ) 

    #----------------------------------------

    def makePSetForEtFilter(self, moduleName):
        """ generates a PSet for the Egamma DQM analyzer for the Et filter """

        return cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(TriggerTypeDefs_cfi.TriggerCluster)
        )

    #----------------------------------------

    def makePSetForOneOEMinusOneOPFilter(self, moduleName):

        return cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(TriggerTypeDefs_cfi.TriggerElectron)
            )

    #----------------------------------------

    def makePSetForPixelMatchFilter(self, moduleName):
        return cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(TriggerTypeDefs_cfi.TriggerCluster)
            )

    #----------------------------------------

    def makePSetForEgammaGenericFilter(self, module, moduleName):

        # example usages of HLTEgammaGenericFilter are:
        #   R9 shape filter                        hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolR9ShapeFilter 
        #   cluster shape filter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolClusterShapeFilter 
        #   Ecal isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TIghterEleIdIsolEcalIsolFilter
        #   H/E filter                             hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHEFilter
        #   HCAL isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHcalIsolFilter

        # the type of object to look for seems to be the
        # same for all uses of HLTEgammaGenericFilter
        theHLTOutputTypes = cms.int32(TriggerTypeDefs_cfi.TriggerCluster)

        # infer the type of filter by the type of the producer which
        # generates the collection used to cut on this
        inputCollectionLabel = module.isoTag.moduleLabel

        inputType = getattr(self.process, inputCollectionLabel).type_()
        # print >> sys.stderr, "inputType=",inputType,moduleName

        #--------------------
        # sanity check: non-isolated path should be produced by the
        # same type of module
        #
        # first check that the non-iso tag is non-empty
        assert(module.nonIsoTag.moduleLabel != "")

        assert(inputType == getattr(self.process, module.nonIsoTag.moduleLabel).type_())

        #--------------------


        # the following cases seem to have identical PSets ?

        #--------------------
        # R9 shape
        #--------------------

        if inputType == 'EgammaHLTR9Producer':
            return cms.PSet(
                PlotBounds = cms.vdouble(0.0, 0.0),
                HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
                IsoCollections = cms.VInputTag(module.isoTag, module.nonIsoTag),
                theHLTOutputTypes = theHLTOutputTypes
                )

        #--------------------
        # cluster shape
        #--------------------
        if inputType == 'EgammaHLTClusterShapeProducer':
            return cms.PSet(
                PlotBounds = cms.vdouble(0.0, 0.0),
                HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
                IsoCollections = cms.VInputTag(module.isoTag, module.nonIsoTag),
                theHLTOutputTypes = theHLTOutputTypes
                )

        #--------------------
        # ecal isolation
        #--------------------
        if inputType == 'EgammaHLTEcalRecIsolationProducer':
            return cms.PSet(
                PlotBounds = cms.vdouble(0.0, 0.0),
                HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
                IsoCollections = cms.VInputTag(module.isoTag, module.nonIsoTag),
                theHLTOutputTypes = theHLTOutputTypes
                )

        #--------------------
        # HCAL isolation and HE
        #--------------------

        if inputType == 'EgammaHLTHcalIsolationProducersRegional':
            return cms.PSet(
                PlotBounds = cms.vdouble(0.0, 0.0),
                HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
                IsoCollections = cms.VInputTag(module.isoTag, module.nonIsoTag),
                theHLTOutputTypes = theHLTOutputTypes
                )
            
        
        raise Exception("can't determine what the HLTEgammaGenericFilter '" + moduleName + "' should do: uses a collection produced by a module of C++ type '" + inputType + "'")

    #----------------------------------------

    def makePSetForElectronGenericFilter(self, module, moduleName):

        # example usages of HLTElectronGenericFilter are:

        # deta filter      hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDetaFilter
        # dphi filter      hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDphiFilter
        # track isolation  hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolTrackIsolFilter

        # the type of object to look for seems to be the
        # same for all uses of HLTEgammaGenericFilter
        theHLTOutputTypes = cms.int32(TriggerTypeDefs_cfi.TriggerElectron)

        # infer the type of filter by the type of the producer which
        # generates the collection used to cut on this
        inputCollectionLabel = module.isoTag.moduleLabel

        inputType = getattr(self.process, inputCollectionLabel).type_()
        # print >> sys.stderr, "inputType=",inputType,moduleName

        # sanity check: non-isolated path should be produced by the
        # same type of module
        assert(inputType == getattr(self.process, module.nonIsoTag.moduleLabel).type_())

        # the following cases seem to have identical PSets ?

        #--------------------
        # deta and dphi filter
        #--------------------

        # note that whether deta or dphi is used is determined from
        # the product instance (not the module label)
        if inputType == 'EgammaHLTElectronDetaDphiProducer':

            return cms.PSet(
                PlotBounds = cms.vdouble(0.0, 0.0),
                HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
                IsoCollections = cms.VInputTag(module.isoTag, module.nonIsoTag),
                theHLTOutputTypes = theHLTOutputTypes
                )

        #--------------------
        # track isolation
        #--------------------

        if inputType == 'EgammaHLTElectronTrackIsolationProducers':

            return cms.PSet(
                PlotBounds = cms.vdouble(0.0, 0.0),
                HLTCollectionLabels = cms.InputTag(moduleName,"",self.processName),
                IsoCollections = cms.VInputTag(module.isoTag, module.nonIsoTag),
                theHLTOutputTypes = theHLTOutputTypes
                )
        raise Exception("can't determine what the HLTElectronGenericFilter '" + moduleName + "' should do: uses a collection produced by a module of C++ type '" + inputType + "'")

    #----------------------------------------

    def getResult(self):
        """ returns the composed analyzer module """
        return self.__result

    #----------------------------------------    

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
if __name__ == "__main__":

    import FWCore.ParameterSet.Config as cms
    process = cms.Process("MYTEST")
    process.load("HLTrigger.Configuration.HLT_GRun_cff")

    moduleMaker = EgammaDQMModuleMaker(process, "HLT_Ele17_SW_TighterEleIdIsol_L1R_v3", 11, 1)

    # print "# ----------------------------------------------------------------------"

    print moduleMaker.getResult().dumpPython()

#----------------------------------------------------------------------

def findEgammaPaths(process):
    """
    returns a dict:

     {
       "singleElectron": [ list of single electron path objects ],
       "doubleElectron": [ list of double electron path objects ],
       "singlePhoton":   [ list of single photon path objects ],
       "doublePhoton":   [ list of double photon path objects ],
     }

     Note that the elements in the lists are path objects, not path names.

     Note also that this is based on the name of the paths using some
     heuristics.
     """

    retval = { "singleElectron": [],
               "doubleElectron": [],
               "singlePhoton":   [],
               "doublePhoton":   [],
               }
    
    for path_name, path in process.paths.items():

        # print "XX",path_name.__class__,path.__class__

        if path_name.startswith("AlCa_"):
            continue

        if path_name.startswith("DQM_"):
            continue

        if not path_name.startswith("HLT_"):
            continue

        if path_name.startswith("HLT_Ele"):
            retval['singleElectron'].append(path)
            continue
        
        if path_name.startswith("HLT_Photon"):
            retval['singlePhoton'].append(path)
            continue

        if path_name.startswith("HLT_DoublePhoton"):
            retval['doublePhoton'].append(path)
            continue

        if path_name.startswith("HLT_DoubleEle"):
            retval['doubleElectron'].append(path)
            continue

    # end of loop over paths
    return retval

#----------------------------------------------------------------------

def getModuleNamesOfPath(path):
    """ returns the names of the modules found in the given path.

    Note that these are not guaranteed to be in any particular
    order.
    """

    # this function could actually call getModulesOfSequence(..)
    # and then produce a set with the unique names of
    # the modules

    import FWCore.ParameterSet.Modules
    class Visitor:

        #----------------------------------------
        def __init__(self):
            self.module_names_found = set()

        #----------------------------------------
        def enter(self,visitee):

            if isinstance(visitee, FWCore.ParameterSet.Modules._Module):
                self.module_names_found.add(visitee.label_())

        #----------------------------------------
        def leave(self,visitee):
            pass

        #----------------------------------------
                
    visitor = Visitor()
    path.visit(visitor)

    return visitor.module_names_found


#----------------------------------------------------------------------
def getCXXTypesOfPath(process, path):
    """ returns the names of (classes) of the C++ types of the modules
    found in the given path (in no particular order) """

    moduleNames = getModuleNamesOfPath(path)

    retval = set()

    for name in moduleNames:

        # skip those modules which are in the path
        # but not in the process object.
        # not understood why we need to do this
        # but seems to cause problems in practice...
        if not hasattr(process,name):
            continue

        module = getattr(process, name)

        retval.add(module.type_())

    return retval

#----------------------------------------------------------------------

def getModulesOfSequence(sequence):
    """ returns the modules found in a sequence.

    Note that a module can appear more than once.
    """

    import FWCore.ParameterSet.Modules
    class Visitor:

        #----------------------------------------
        def __init__(self):
            self.modules_found = []

        #----------------------------------------
        def enter(self,visitee):

            if isinstance(visitee, FWCore.ParameterSet.Modules._Module):
                self.modules_found.append(visitee)

        #----------------------------------------
        def leave(self,visitee):
            pass

        #----------------------------------------
                
    visitor = Visitor()
    sequence.visitNode(visitor)

    return visitor.modules_found


#----------------------------------------------------------------------

from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import patDiscriminationByIsolationMVArun2v1raw, patDiscriminationByIsolationMVArun2v1
import os
import re
import six

class TauIDEmbedder(object):
    """class to rerun the tau seq and acces trainings from the database"""
    availableDiscriminators = [
        "2017v1", "2017v2", "newDM2017v2", "dR0p32017v2", "2016v1", "newDM2016v1",
        "deepTau2017v1", "deepTau2017v2", "deepTau2017v2p1",
        "DPFTau_2016_v0", "DPFTau_2016_v1",
        "againstEle2018",
        "newDMPhase2v1"
    ]

    def __init__(self, process, debug = False,
                 updatedTauName = "slimmedTausNewID",
                 toKeep =  ["deepTau2017v2p1"],
                 tauIdDiscrMVA_trainings_run2_2017 = { 'tauIdMVAIsoDBoldDMwLT2017' : "tauIdMVAIsoDBoldDMwLT2017", },
                 tauIdDiscrMVA_WPs_run2_2017 = {
                    'tauIdMVAIsoDBoldDMwLT2017' : {
                        'Eff95' : "DBoldDMwLTEff95",
                        'Eff90' : "DBoldDMwLTEff90",
                        'Eff80' : "DBoldDMwLTEff80",
                        'Eff70' : "DBoldDMwLTEff70",
                        'Eff60' : "DBoldDMwLTEff60",
                        'Eff50' : "DBoldDMwLTEff50",
                        'Eff40' : "DBoldDMwLTEff40"
                    }
                 },
                 tauIdDiscrMVA_2017_version = "v1",
                 conditionDB = "" # preparational DB: 'frontier://FrontierPrep/CMS_CONDITIONS'
                 ):
        super(TauIDEmbedder, self).__init__()
        self.process = process
        self.debug = debug
        self.updatedTauName = updatedTauName
        self.process.load('RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi')
        if len(conditionDB) != 0:
            self.process.CondDBTauConnection.connect = cms.string(conditionDB)
            self.process.loadRecoTauTagMVAsFromPrepDB.connect = cms.string(conditionDB)
            # if debug:
            # 	print self.process.CondDBTauConnection.connect
            # 	print dir(self.process.loadRecoTauTagMVAsFromPrepDB)
            # 	print self.process.loadRecoTauTagMVAsFromPrepDB.parameterNames_

        self.tauIdDiscrMVA_trainings_run2_2017 = tauIdDiscrMVA_trainings_run2_2017
        self.tauIdDiscrMVA_WPs_run2_2017 = tauIdDiscrMVA_WPs_run2_2017
        self.tauIdDiscrMVA_2017_version = tauIdDiscrMVA_2017_version
        for discr in toKeep:
            if discr not in TauIDEmbedder.availableDiscriminators:
                raise RuntimeError('TauIDEmbedder: discriminator "{}" is not supported'.format(discr))
        self.toKeep = toKeep

    
    @staticmethod
    def get_cmssw_version(debug = False):
        """returns 'CMSSW_X_Y_Z'"""
        cmssw_version = os.environ["CMSSW_VERSION"]
        if debug: print ("get_cmssw_version:", cmssw_version)
        return cmssw_version

    @classmethod
    def get_cmssw_version_number(klass, debug = False):
        """returns '(release, subversion, patch)' (without 'CMSSW_')"""
        v = klass.get_cmssw_version().split("CMSSW_")[1].split("_")[0:3]
        if debug: print ("get_cmssw_version_number:", v)
        try:
            patch = int(v[2])
        except:
            patch = -1
        return int(v[0]), int(v[1]), patch

    @staticmethod
    def versionToInt(release=9, subversion=4, patch=0, debug = False):
        version = release * 10000 + subversion * 100 + patch + 1 # shifted by one to account for pre-releases.
        if debug: print ("versionToInt:", version)
        return version


    @classmethod
    def is_above_cmssw_version(klass, release=9, subversion=4, patch=0, debug = False):
        split_cmssw_version = klass.get_cmssw_version_number()
        if klass.versionToInt(release, subversion, patch) > klass.versionToInt(split_cmssw_version[0], split_cmssw_version[1], split_cmssw_version[2]):
            if debug: print ("is_above_cmssw_version:", False)
            return False
        else:
            if debug: print ("is_above_cmssw_version:", True)
            return True

    def tauIDMVAinputs(self, module, wp):
        return cms.PSet(inputTag = cms.InputTag(module), workingPointIndex = cms.int32(-1 if wp=="raw" else -2 if wp=="category" else getattr(self.process, module).workingPoints.index(wp)))

    def loadMVA_WPs_run2_2017(self):
        if self.debug: print ("loadMVA_WPs_run2_2017: performed")
        global cms
        for training, gbrForestName in self.tauIdDiscrMVA_trainings_run2_2017.items():

            self.process.loadRecoTauTagMVAsFromPrepDB.toGet.append(
                cms.PSet(
                    record = cms.string('GBRWrapperRcd'),
                    tag = cms.string("RecoTauTag_%s%s" % (gbrForestName, self.tauIdDiscrMVA_2017_version)),
                    label = cms.untracked.string("RecoTauTag_%s%s" % (gbrForestName, self.tauIdDiscrMVA_2017_version))
                )
            )

            for WP in self.tauIdDiscrMVA_WPs_run2_2017[training].keys():
                self.process.loadRecoTauTagMVAsFromPrepDB.toGet.append(
                    cms.PSet(
                        record = cms.string('PhysicsTGraphPayloadRcd'),
                        tag = cms.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, self.tauIdDiscrMVA_2017_version, WP)),
                        label = cms.untracked.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, self.tauIdDiscrMVA_2017_version, WP))
                    )
                )

            self.process.loadRecoTauTagMVAsFromPrepDB.toGet.append(
                cms.PSet(
                    record = cms.string('PhysicsTFormulaPayloadRcd'),
                    tag = cms.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, self.tauIdDiscrMVA_2017_version)),
                    label = cms.untracked.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, self.tauIdDiscrMVA_2017_version))
                )
            )

    def runTauID(self):
        self.process.rerunMvaIsolationTask = cms.Task()
        self.process.rerunMvaIsolationSequence = cms.Sequence()
        tauIDSources = cms.PSet()

        # rerun the seq to obtain the 2017 nom training with 0.5 iso cone, old DM, ptph>1, trained on 2017MCv1
        if "2017v1" in self.toKeep:
            self.tauIdDiscrMVA_2017_version = "v1"
            self.tauIdDiscrMVA_trainings_run2_2017 = {
                'tauIdMVAIsoDBoldDMwLT2017' : "tauIdMVAIsoDBoldDMwLT2017",
            }
            self.tauIdDiscrMVA_WPs_run2_2017 = {
                'tauIdMVAIsoDBoldDMwLT2017' : {
                    'Eff95' : "DBoldDMwLTEff95",
                    'Eff90' : "DBoldDMwLTEff90",
                    'Eff80' : "DBoldDMwLTEff80",
                    'Eff70' : "DBoldDMwLTEff70",
                    'Eff60' : "DBoldDMwLTEff60",
                    'Eff50' : "DBoldDMwLTEff50",
                    'Eff40' : "DBoldDMwLTEff40"
                }
            }
            # update the list of available in DB samples
            if not self.is_above_cmssw_version(9, 4, 4, self.debug):
                if self.debug: print ("runTauID: not is_above_cmssw_version(9, 4, 4). Will update the list of available in DB samples to access 2017v1")
                self.loadMVA_WPs_run2_2017()

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1"),#RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1 writeTauIdDiscrMVAs
                mvaOpt = cms.string("DBoldDMwLTwGJ"),
                verbosity = cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1 = patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1raw'),
                loadMVAfromDB = cms.bool(True),
                mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
                mapping = cms.VPSet(
                    cms.PSet(
                        category = cms.uint32(0),
                        cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1"), #writeTauIdDiscrWPs
                        variable = cms.string("pt"),
                    )
                ),
                workingPoints = cms.vstring(
                    "_WPEff95",
                    "_WPEff90",
                    "_WPEff80",
                    "_WPEff70",
                    "_WPEff60",
                    "_WPEff50",
                    "_WPEff40"
                )
            )

            self.rerunIsolationOldDMMVArun2017v1Task =  cms.Task(
                self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1raw,
                self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1
            )
            self.process.rerunMvaIsolationTask.add(self.rerunIsolationOldDMMVArun2017v1Task)
            self.process.rerunMvaIsolationSequence += cms.Sequence(self.rerunIsolationOldDMMVArun2017v1Task)

            tauIDSources.byIsolationMVArun2017v1DBoldDMwLTraw2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v1", "raw")
            tauIDSources.byVVLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v1", "_WPEff95")
            tauIDSources.byVLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v1", "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v1", "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v1", "_WPEff70")
            tauIDSources.byTightIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v1", "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v1", "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v1", "_WPEff40")


        if "2017v2" in self.toKeep:
            self.tauIdDiscrMVA_2017_version = "v2"
            self.tauIdDiscrMVA_trainings_run2_2017 = {
                'tauIdMVAIsoDBoldDMwLT2017' : "tauIdMVAIsoDBoldDMwLT2017",
            }
            self.tauIdDiscrMVA_WPs_run2_2017 = {
                'tauIdMVAIsoDBoldDMwLT2017' : {
                    'Eff95' : "DBoldDMwLTEff95",
                    'Eff90' : "DBoldDMwLTEff90",
                    'Eff80' : "DBoldDMwLTEff80",
                    'Eff70' : "DBoldDMwLTEff70",
                    'Eff60' : "DBoldDMwLTEff60",
                    'Eff50' : "DBoldDMwLTEff50",
                    'Eff40' : "DBoldDMwLTEff40"
                }
            }

            if not self.is_above_cmssw_version(9, 4, 5, self.debug):
                if self.debug: print ("runTauID: not is_above_cmssw_version(9, 4, 5). Will update the list of available in DB samples to access 2017v2")
                self.loadMVA_WPs_run2_2017()

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2"),#RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1 writeTauIdDiscrMVAs
                mvaOpt = cms.string("DBoldDMwLTwGJ"),
                verbosity = cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2 = patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2raw'),
                loadMVAfromDB = cms.bool(True),
                mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
                mapping = cms.VPSet(
                    cms.PSet(
                        category = cms.uint32(0),
                        cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2"), #writeTauIdDiscrWPs
                        variable = cms.string("pt"),
                    )
                ),
                workingPoints = cms.vstring(
                    "_WPEff95",
                    "_WPEff90",
                    "_WPEff80",
                    "_WPEff70",
                    "_WPEff60",
                    "_WPEff50",
                    "_WPEff40"
                ),
                verbosity = cms.int32(0)
            )

            self.rerunIsolationOldDMMVArun2017v2Task = cms.Task(
                self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2raw,
                self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2
            )
            self.process.rerunMvaIsolationTask.add(self.rerunIsolationOldDMMVArun2017v2Task)
            self.process.rerunMvaIsolationSequence += cms.Sequence(self.rerunIsolationOldDMMVArun2017v2Task)

            tauIDSources.byIsolationMVArun2017v2DBoldDMwLTraw2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v2", "raw")
            tauIDSources.byVVLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v2", "_WPEff95")
            tauIDSources.byVLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v2", "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v2", "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v2", "_WPEff70")
            tauIDSources.byTightIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v2", "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v2", "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2017v2", "_WPEff40")

        if "newDM2017v2" in self.toKeep:
            self.tauIdDiscrMVA_2017_version = "v2"
            self.tauIdDiscrMVA_trainings_run2_2017 = {
                'tauIdMVAIsoDBnewDMwLT2017' : "tauIdMVAIsoDBnewDMwLT2017",
            }
            self.tauIdDiscrMVA_WPs_run2_2017 = {
                'tauIdMVAIsoDBnewDMwLT2017' : {
                    'Eff95' : "DBnewDMwLTEff95",
                    'Eff90' : "DBnewDMwLTEff90",
                    'Eff80' : "DBnewDMwLTEff80",
                    'Eff70' : "DBnewDMwLTEff70",
                    'Eff60' : "DBnewDMwLTEff60",
                    'Eff50' : "DBnewDMwLTEff50",
                    'Eff40' : "DBnewDMwLTEff40"
                }
            }

            if not self.is_above_cmssw_version(9, 4, 5, self.debug):
                if self.debug: print ("runTauID: not is_above_cmssw_version(9, 4, 5). Will update the list of available in DB samples to access newDM2017v2")
                self.loadMVA_WPs_run2_2017()

            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2"),#RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1 writeTauIdDiscrMVAs
                mvaOpt = cms.string("DBnewDMwLTwGJ"),
                verbosity = cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2 = patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2raw'),
                loadMVAfromDB = cms.bool(True),
                mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
                mapping = cms.VPSet(
                    cms.PSet(
                        category = cms.uint32(0),
                        cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2"), #writeTauIdDiscrWPs
                        variable = cms.string("pt"),
                    )
                ),
                workingPoints = cms.vstring(
                    "_WPEff95",
                    "_WPEff90",
                    "_WPEff80",
                    "_WPEff70",
                    "_WPEff60",
                    "_WPEff50",
                    "_WPEff40"
                ),
                verbosity = cms.int32(0)
            )

            self.rerunIsolationNewDMMVArun2017v2Task = cms.Task(
                self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2raw,
                self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2
            )
            self.process.rerunMvaIsolationTask.add(self.rerunIsolationNewDMMVArun2017v2Task)
            self.process.rerunMvaIsolationSequence += cms.Sequence(self.rerunIsolationNewDMMVArun2017v2Task)

            tauIDSources.byIsolationMVArun2017v2DBnewDMwLTraw2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2017v2", "raw")
            tauIDSources.byVVLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2017v2", "_WPEff95")
            tauIDSources.byVLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2017v2", "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2017v2", "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2017v2", "_WPEff70")
            tauIDSources.byTightIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2017v2", "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2017v2", "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2017v2", "_WPEff40")

        if "dR0p32017v2" in self.toKeep:
            self.tauIdDiscrMVA_2017_version = "v2"
            self.tauIdDiscrMVA_trainings_run2_2017 = {
                'tauIdMVAIsoDBoldDMdR0p3wLT2017' : "tauIdMVAIsoDBoldDMdR0p3wLT2017",
            }
            self.tauIdDiscrMVA_WPs_run2_2017 = {
                'tauIdMVAIsoDBoldDMdR0p3wLT2017' : {
                    'Eff95' : "DBoldDMdR0p3wLTEff95",
                    'Eff90' : "DBoldDMdR0p3wLTEff90",
                    'Eff80' : "DBoldDMdR0p3wLTEff80",
                    'Eff70' : "DBoldDMdR0p3wLTEff70",
                    'Eff60' : "DBoldDMdR0p3wLTEff60",
                    'Eff50' : "DBoldDMdR0p3wLTEff50",
                    'Eff40' : "DBoldDMdR0p3wLTEff40"
                }
            }

            if not self.is_above_cmssw_version(9, 4, 5, self.debug):
                if self.debug: print ("runTauID: not is_above_cmssw_version(9, 4, 5). Will update the list of available in DB samples to access dR0p32017v2")
                self.loadMVA_WPs_run2_2017()

            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2"),
                mvaOpt = cms.string("DBoldDMwLTwGJ"),
                srcChargedIsoPtSum = cms.string('chargedIsoPtSumdR03'),
                srcFootprintCorrection = cms.string('footprintCorrectiondR03'),
                srcNeutralIsoPtSum = cms.string('neutralIsoPtSumdR03'),
                srcPhotonPtSumOutsideSignalCone = cms.string('photonPtSumOutsideSignalConedR03'),
                verbosity = cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2= patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2raw'),
                loadMVAfromDB = cms.bool(True),
                mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
                mapping = cms.VPSet(
                    cms.PSet(
                        category = cms.uint32(0),
                        cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2"), #writeTauIdDiscrWPs
                        variable = cms.string("pt"),
                    )
                ),
                workingPoints = cms.vstring(
                    "_WPEff95",
                    "_WPEff90",
                    "_WPEff80",
                    "_WPEff70",
                    "_WPEff60",
                    "_WPEff50",
                    "_WPEff40"
                ),
                verbosity = cms.int32(0)
            )

            self.rerunIsolationOldDMdR0p3MVArun2017v2Task = cms.Task(
                self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2raw,
                self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2
            )
            self.process.rerunMvaIsolationTask.add(self.rerunIsolationOldDMdR0p3MVArun2017v2Task)
            self.process.rerunMvaIsolationSequence += cms.Sequence(self.rerunIsolationOldDMdR0p3MVArun2017v2Task)

            tauIDSources.byIsolationMVArun2017v2DBoldDMdR0p3wLTraw2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2", "raw")
            tauIDSources.byVVLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2", "_WPEff95")
            tauIDSources.byVLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2", "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2", "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2", "_WPEff70")
            tauIDSources.byTightIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2", "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2", "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2", "_WPEff40")

        # 2016 training strategy(v2) - essentially the same as 2017 training strategy (v1), trained on 2016MC, old DM - currently not implemented in the tau sequence of any release
        # self.process.rerunDiscriminationByIsolationOldDMMVArun2v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
        #     PATTauProducer = cms.InputTag('slimmedTaus'),
        #     Prediscriminants = noPrediscriminants,
        #     loadMVAfromDB = cms.bool(True),
        #     mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v2"),#RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1 writeTauIdDiscrMVAs
        #     mvaOpt = cms.string("DBoldDMwLTwGJ"),
        #     verbosity = cms.int32(0)
        # )
        # #
        # self.process.rerunDiscriminationByIsolationOldDMMVArun2v2VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
        #     PATTauProducer = cms.InputTag('slimmedTaus'),
        #     Prediscriminants = noPrediscriminants,
        #     toMultiplex = cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v2raw'),
        #     key = cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v2raw:category'),#?
        #     loadMVAfromDB = cms.bool(True),
        #     mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v2_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
        #     mapping = cms.VPSet(
        #         cms.PSet(
        #             category = cms.uint32(0),
        #             cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v2_WPEff90"), #writeTauIdDiscrWPs
        #             variable = cms.string("pt"),
        #         )
        #     )
        # )

        # 2016 training strategy(v1), trained on 2016MC, old DM
        if "2016v1" in self.toKeep:
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1"),
                mvaOpt = cms.string("DBoldDMwLT"),
                verbosity = cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1 = patDiscriminationByIsolationMVArun2v1.clone(
                    PATTauProducer = cms.InputTag('slimmedTaus'),
                    Prediscriminants = noPrediscriminants,
                    toMultiplex = cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1raw'),
                    loadMVAfromDB = cms.bool(True),
                    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_mvaOutput_normalization"),
                    mapping = cms.VPSet(
                        cms.PSet(
                            category = cms.uint32(0),
                            cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1"),
                            variable = cms.string("pt"),
                        )
                    ),
                    workingPoints = cms.vstring(
                        "_WPEff90",
                        "_WPEff80",
                        "_WPEff70",
                        "_WPEff60",
                        "_WPEff50",
                        "_WPEff40"
                    )
                )

            self.rerunIsolationOldDMMVArun2016v1Task = cms.Task(
                self.process.rerunDiscriminationByIsolationOldDMMVArun2v1raw,
                self.process.rerunDiscriminationByIsolationOldDMMVArun2v1
            )
            self.process.rerunMvaIsolationTask.add(self.rerunIsolationOldDMMVArun2016v1Task)
            self.process.rerunMvaIsolationSequence += cms.Sequence(self.rerunIsolationOldDMMVArun2016v1Task)

            tauIDSources.byIsolationMVArun2v1DBoldDMwLTraw2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2v1", "raw")
            tauIDSources.byVLooseIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2v1", "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2v1", "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2v1", "_WPEff70")
            tauIDSources.byTightIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2v1", "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2v1", "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationOldDMMVArun2v1", "_WPEff40")

        # 2016 training strategy(v1), trained on 2016MC, new DM
        if "newDM2016v1" in self.toKeep:
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1"),
                mvaOpt = cms.string("DBnewDMwLT"),
                verbosity = cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1 = patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1raw'),
                loadMVAfromDB = cms.bool(True),
                mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_mvaOutput_normalization"),
                mapping = cms.VPSet(
                    cms.PSet(
                        category = cms.uint32(0),
                        cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_WPEff90"),
                        variable = cms.string("pt"),
                    )
                ),
                workingPoints = cms.vstring(
                    "_WPEff90",
                    "_WPEff80",
                    "_WPEff70",
                    "_WPEff60",
                    "_WPEff50",
                    "_WPEff40"
                )
            )

            self.rerunIsolationNewDMMVArun2016v1Task = cms.Task(
                self.process.rerunDiscriminationByIsolationNewDMMVArun2v1raw,
                self.process.rerunDiscriminationByIsolationNewDMMVArun2v1
            )
            self.process.rerunMvaIsolationTask.add(self.rerunIsolationNewDMMVArun2016v1Task)
            self.process.rerunMvaIsolationSequence += cms.Sequence(self.rerunIsolationNewDMMVArun2016v1Task)

            tauIDSources.byIsolationMVArun2v1DBnewDMwLTraw2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2v1", "raw")
            tauIDSources.byVLooseIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2v1", "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2v1", "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2v1", "_WPEff70")
            tauIDSources.byTightIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2v1", "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2v1", "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs("rerunDiscriminationByIsolationNewDMMVArun2v1", "_WPEff40")

        if "deepTau2017v1" in self.toKeep:
            if self.debug: print ("Adding DeepTau IDs")

            workingPoints_ = {
                "e": {
                    "VVVLoose" : 0.96424,
                    "VVLoose" : 0.98992,
                    "VLoose" : 0.99574,
                    "Loose": 0.99831,
                    "Medium": 0.99868,
                    "Tight": 0.99898,
                    "VTight": 0.99911,
                    "VVTight": 0.99918
                },
                "mu": {
                    "VVVLoose" : 0.959619,
                    "VVLoose" : 0.997687,
                    "VLoose" : 0.999392,
                    "Loose": 0.999755,
                    "Medium": 0.999854,
                    "Tight": 0.999886,
                    "VTight": 0.999944,
                    "VVTight": 0.9999971
                },

                "jet": {
                    "VVVLoose" : 0.5329,
                    "VVLoose" : 0.7645,
                    "VLoose" : 0.8623,
                    "Loose": 0.9140,
                    "Medium": 0.9464,
                    "Tight": 0.9635,
                    "VTight": 0.9760,
                    "VVTight": 0.9859
                }
            }
            file_names = ['RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v1_20L1024N_quantized.pb']
            self.process.deepTau2017v1 = cms.EDProducer("DeepTauId",
                electrons              = cms.InputTag('slimmedElectrons'),
                muons                  = cms.InputTag('slimmedMuons'),
                taus                   = cms.InputTag('slimmedTaus'),
                pfcands                = cms.InputTag('packedPFCandidates'),
                vertices               = cms.InputTag('offlineSlimmedPrimaryVertices'),
                rho                    = cms.InputTag('fixedGridRhoAll'),
                graph_file             = cms.vstring(file_names),
                mem_mapped             = cms.bool(False),
                version                = cms.uint32(self.getDeepTauVersion(file_names[0])[1]),
                debug_level            = cms.int32(0),
                disable_dxy_pca        = cms.bool(False)
            )

            self.processDeepProducer('deepTau2017v1', tauIDSources, workingPoints_)

            self.process.rerunMvaIsolationTask.add(self.process.deepTau2017v1)
            self.process.rerunMvaIsolationSequence += self.process.deepTau2017v1

        if "deepTau2017v2" in self.toKeep:
            if self.debug: print ("Adding DeepTau IDs")

            workingPoints_ = {
                "e": {
                    "VVVLoose": 0.0630386,
                    "VVLoose": 0.1686942,
                    "VLoose": 0.3628130,
                    "Loose": 0.6815435,
                    "Medium": 0.8847544,
                    "Tight": 0.9675541,
                    "VTight": 0.9859251,
                    "VVTight": 0.9928449,
                },
                "mu": {
                    "VLoose": 0.1058354,
                    "Loose": 0.2158633,
                    "Medium": 0.5551894,
                    "Tight": 0.8754835,
                },
                "jet": {
                    "VVVLoose": 0.2599605,
                    "VVLoose": 0.4249705,
                    "VLoose": 0.5983682,
                    "Loose": 0.7848675,
                    "Medium": 0.8834768,
                    "Tight": 0.9308689,
                    "VTight": 0.9573137,
                    "VVTight": 0.9733927,
                },
            }

            file_names = [
                'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.pb',
                'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_inner.pb',
                'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_outer.pb',
            ]
            self.process.deepTau2017v2 = cms.EDProducer("DeepTauId",
                electrons              = cms.InputTag('slimmedElectrons'),
                muons                  = cms.InputTag('slimmedMuons'),
                taus                   = cms.InputTag('slimmedTaus'),
                pfcands                = cms.InputTag('packedPFCandidates'),
                vertices               = cms.InputTag('offlineSlimmedPrimaryVertices'),
                rho                    = cms.InputTag('fixedGridRhoAll'),
                graph_file             = cms.vstring(file_names),
                mem_mapped             = cms.bool(False),
                version                = cms.uint32(self.getDeepTauVersion(file_names[0])[1]),
                debug_level            = cms.int32(0),
                disable_dxy_pca        = cms.bool(False)
            )

            self.processDeepProducer('deepTau2017v2', tauIDSources, workingPoints_)

            self.process.rerunMvaIsolationTask.add(self.process.deepTau2017v2)
            self.process.rerunMvaIsolationSequence += self.process.deepTau2017v2

        if "deepTau2017v2p1" in self.toKeep:
            if self.debug: print ("Adding DeepTau IDs")

            workingPoints_ = {
                "e": {
                    "VVVLoose": 0.0630386,
                    "VVLoose": 0.1686942,
                    "VLoose": 0.3628130,
                    "Loose": 0.6815435,
                    "Medium": 0.8847544,
                    "Tight": 0.9675541,
                    "VTight": 0.9859251,
                    "VVTight": 0.9928449,
                },
                "mu": {
                    "VLoose": 0.1058354,
                    "Loose": 0.2158633,
                    "Medium": 0.5551894,
                    "Tight": 0.8754835,
                },
                "jet": {
                    "VVVLoose": 0.2599605,
                    "VVLoose": 0.4249705,
                    "VLoose": 0.5983682,
                    "Loose": 0.7848675,
                    "Medium": 0.8834768,
                    "Tight": 0.9308689,
                    "VTight": 0.9573137,
                    "VVTight": 0.9733927,
                },
            }

            file_names = [
                'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.pb',
                'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_inner.pb',
                'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_outer.pb',
            ]
            self.process.deepTau2017v2p1 = cms.EDProducer("DeepTauId",
                electrons                = cms.InputTag('slimmedElectrons'),
                muons                    = cms.InputTag('slimmedMuons'),
                taus                     = cms.InputTag('slimmedTaus'),
                pfcands                  = cms.InputTag('packedPFCandidates'),
                vertices                 = cms.InputTag('offlineSlimmedPrimaryVertices'),
                rho                      = cms.InputTag('fixedGridRhoAll'),
                graph_file               = cms.vstring(file_names),
                mem_mapped               = cms.bool(False),
                version                  = cms.uint32(self.getDeepTauVersion(file_names[0])[1]),
                debug_level              = cms.int32(0),
                disable_dxy_pca          = cms.bool(True)
            )

            self.processDeepProducer('deepTau2017v2p1', tauIDSources, workingPoints_)

            self.process.rerunMvaIsolationTask.add(self.process.deepTau2017v2p1)
            self.process.rerunMvaIsolationSequence += self.process.deepTau2017v2p1

        if "DPFTau_2016_v0" in self.toKeep:
            if self.debug: print ("Adding DPFTau isolation (v0)")

            workingPoints_ = {
                "all": {
                    "Tight" : "if(decayMode == 0) return (0.898328 - 0.000160992 * pt);" + \
                              "if(decayMode == 1) return (0.910138 - 0.000229923 * pt);" + \
                              "if(decayMode == 10) return (0.873958 - 0.0002328 * pt);" + \
                              "return 99.0;"
                    #"Tight" : "? decayMode == 0 ? (0.898328 - 0.000160992 * pt) : " +
                    #          "(? decayMode == 1 ? 0.910138 - 0.000229923 * pt : " +
                    #          "(? decayMode == 10 ? (0.873958 - 0.0002328 * pt) : 1))"
                    # "Tight" : "(decayMode == 0) * (0.898328 - 0.000160992 * pt) + \
                    #            (decayMode == 1) * (0.910138 - 0.000229923 * pt) + \
                    #            (decayMode == 10) * (0.873958 - 0.0002328 * pt) "
                }
            }
            file_names = [ 'RecoTauTag/TrainingFiles/data/DPFTauId/DPFIsolation_2017v0_quantized.pb' ]
            self.process.dpfTau2016v0 = cms.EDProducer("DPFIsolation",
                pfcands     = cms.InputTag('packedPFCandidates'),
                taus        = cms.InputTag('slimmedTaus'),
                vertices    = cms.InputTag('offlineSlimmedPrimaryVertices'),
                graph_file  = cms.vstring(file_names),
                version     = cms.uint32(self.getDpfTauVersion(file_names[0])),
                mem_mapped  = cms.bool(False)
            )

            self.processDeepProducer('dpfTau2016v0', tauIDSources, workingPoints_)

            self.process.rerunMvaIsolationTask.add(self.process.dpfTau2016v0)
            self.process.rerunMvaIsolationSequence += self.process.dpfTau2016v0


        if "DPFTau_2016_v1" in self.toKeep:
            print ("Adding DPFTau isolation (v1)")
            print ("WARNING: WPs are not defined for DPFTau_2016_v1")
            print ("WARNING: The score of DPFTau_2016_v1 is inverted: i.e. for Sig->0, for Bkg->1 with -1 for undefined input (preselection not passed).")

            workingPoints_ = {
                "all": {"Tight" : 0.123} #FIXME: define WP
            }

            file_names = [ 'RecoTauTag/TrainingFiles/data/DPFTauId/DPFIsolation_2017v1_quantized.pb' ]
            self.process.dpfTau2016v1 = cms.EDProducer("DPFIsolation",
                pfcands     = cms.InputTag('packedPFCandidates'),
                taus        = cms.InputTag('slimmedTaus'),
                vertices    = cms.InputTag('offlineSlimmedPrimaryVertices'),
                graph_file  = cms.vstring(file_names),
                version     = cms.uint32(self.getDpfTauVersion(file_names[0])),
                mem_mapped  = cms.bool(False)
            )

            self.processDeepProducer('dpfTau2016v1', tauIDSources, workingPoints_)

            self.process.rerunMvaIsolationTask.add(self.process.dpfTau2016v1)
            self.process.rerunMvaIsolationSequence += self.process.dpfTau2016v1

        if "againstEle2018" in self.toKeep:
            antiElectronDiscrMVA6_version = "MVA6v3_noeveto"
            ### Define new anti-e discriminants
            ## Raw
            from RecoTauTag.RecoTau.patTauDiscriminationAgainstElectronMVA6_cfi import patTauDiscriminationAgainstElectronMVA6
            self.process.patTauDiscriminationByElectronRejectionMVA62018Raw = patTauDiscriminationAgainstElectronMVA6.clone(
                PATTauProducer = cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants, #already selected for MiniAOD
                srcElectrons = cms.InputTag('slimmedElectrons'),
                vetoEcalCracks = cms.bool(False), #keep taus in EB-EE cracks
                mvaName_NoEleMatch_wGwoGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_wGwoGSF_BL',
                mvaName_NoEleMatch_wGwoGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_wGwoGSF_EC',
                mvaName_NoEleMatch_woGwoGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_woGwoGSF_BL',
                mvaName_NoEleMatch_woGwoGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_woGwoGSF_EC',
                mvaName_wGwGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_wGwGSF_BL',
                mvaName_wGwGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_wGwGSF_EC',
                mvaName_woGwGSF_BL = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_woGwGSF_BL',
                mvaName_woGwGSF_EC = 'RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_woGwGSF_EC'
            )
            ## WPs
            from RecoTauTag.RecoTau.PATTauDiscriminantCutMultiplexer_cfi import patTauDiscriminantCutMultiplexer
            self.process.patTauDiscriminationByElectronRejectionMVA62018 = patTauDiscriminantCutMultiplexer.clone(
                PATTauProducer = self.process.patTauDiscriminationByElectronRejectionMVA62018Raw.PATTauProducer,
                Prediscriminants = self.process.patTauDiscriminationByElectronRejectionMVA62018Raw.Prediscriminants,
                toMultiplex = cms.InputTag("patTauDiscriminationByElectronRejectionMVA62018Raw"),
                mapping = cms.VPSet(
                    cms.PSet(
                        category = cms.uint32(0),
                        cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_woGwoGSF_BL'),
                        variable = cms.string('pt')
                    ),
                    cms.PSet(
                        category = cms.uint32(2),
                        cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_wGwoGSF_BL'),
                        variable = cms.string('pt')
                    ),
                    cms.PSet(
                        category = cms.uint32(5),
                        cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_woGwGSF_BL'),
                        variable = cms.string('pt')
                    ),
                    cms.PSet(
                        category = cms.uint32(7),
                        cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_wGwGSF_BL'),
                        variable = cms.string('pt')
                    ),
                    cms.PSet(
                        category = cms.uint32(8),
                        cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_woGwoGSF_EC'),
                        variable = cms.string('pt')
                    ),
                    cms.PSet(
                        category = cms.uint32(10),
                        cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_wGwoGSF_EC'),
                        variable = cms.string('pt')
                    ),
                    cms.PSet(
                        category = cms.uint32(13),
                        cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_woGwGSF_EC'),
                        variable = cms.string('pt')
                    ),
                    cms.PSet(
                        category = cms.uint32(15),
                        cut = cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_wGwGSF_EC'),
                        variable = cms.string('pt')
                    )
                ),
                workingPoints = cms.vstring(
                    "_WPeff98",
                    "_WPeff90",
                    "_WPeff80",
                    "_WPeff70",
                    "_WPeff60"
                )
            )
            ### Put all new anti-e discrminats to a sequence
            self.process.patTauDiscriminationByElectronRejectionMVA62018Task = cms.Task(
                self.process.patTauDiscriminationByElectronRejectionMVA62018Raw,
                self.process.patTauDiscriminationByElectronRejectionMVA62018
            )
            self.process.patTauDiscriminationByElectronRejectionMVA62018Seq = cms.Sequence(self.process.patTauDiscriminationByElectronRejectionMVA62018Task)
            self.process.rerunMvaIsolationTask.add(self.process.patTauDiscriminationByElectronRejectionMVA62018Task)
            self.process.rerunMvaIsolationSequence += self.process.patTauDiscriminationByElectronRejectionMVA62018Seq

            _againstElectronTauIDSources = cms.PSet(
                againstElectronMVA6Raw2018 = self.tauIDMVAinputs("patTauDiscriminationByElectronRejectionMVA62018", "raw"),
                againstElectronMVA6category2018 = self.tauIDMVAinputs("patTauDiscriminationByElectronRejectionMVA62018", "category"),
                againstElectronVLooseMVA62018 = self.tauIDMVAinputs("patTauDiscriminationByElectronRejectionMVA62018", "_WPeff98"),
                againstElectronLooseMVA62018 = self.tauIDMVAinputs("patTauDiscriminationByElectronRejectionMVA62018", "_WPeff90"),
                againstElectronMediumMVA62018 = self.tauIDMVAinputs("patTauDiscriminationByElectronRejectionMVA62018", "_WPeff80"),
                againstElectronTightMVA62018 = self.tauIDMVAinputs("patTauDiscriminationByElectronRejectionMVA62018", "_WPeff70"),
                againstElectronVTightMVA62018 = self.tauIDMVAinputs("patTauDiscriminationByElectronRejectionMVA62018", "_WPeff60")
            )
            _tauIDSourcesWithAgainistEle = cms.PSet(
                tauIDSources.clone(),
                _againstElectronTauIDSources
            )
            tauIDSources =_tauIDSourcesWithAgainistEle.clone()

        if "newDMPhase2v1" in self.toKeep:
            if self.debug: print ("Adding newDMPhase2v1 ID")
            def tauIDMVAinputs(module, wp):
                return cms.PSet(inputTag = cms.InputTag(module), workingPointIndex = cms.int32(-1 if wp=="raw" else -2 if wp=="category" else getattr(self.process, module).workingPoints.index(wp)))
            self.process.rerunDiscriminationByIsolationMVADBnewDMwLTPhase2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = 'slimmedTaus',
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = True,
                mvaName = 'RecoTauTag_tauIdMVAIsoPhase2',
                mvaOpt = 'DBnewDMwLTwGJPhase2',
                verbosity = 0
            )

            self.process.rerunDiscriminationByIsolationMVADBnewDMwLTPhase2 = patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = 'slimmedTaus',
                Prediscriminants = noPrediscriminants,
                toMultiplex = 'rerunDiscriminationByIsolationMVADBnewDMwLTPhase2raw',
                loadMVAfromDB = True,
                mvaOutput_normalization = 'RecoTauTag_tauIdMVAIsoPhase2_mvaOutput_normalization',
                mapping = cms.VPSet(
                    cms.PSet(
                        category = cms.uint32(0),
                        cut = cms.string("RecoTauTag_tauIdMVAIsoPhase2"),
                        variable = cms.string("pt"),
                    )
                ),
                workingPoints = cms.vstring(
                    "_WPEff95",
                    "_WPEff90",
                    "_WPEff80",
                    "_WPEff70",
                    "_WPEff60",
                    "_WPEff50",
                    "_WPEff40"
                )
            )
            self.process.rerunIsolationMVADBnewDMwLTPhase2Task = cms.Task(
                self.process.rerunDiscriminationByIsolationMVADBnewDMwLTPhase2raw,
                self.process.rerunDiscriminationByIsolationMVADBnewDMwLTPhase2
            )
            self.process.rerunMvaIsolationTask.add(self.process.rerunIsolationMVADBnewDMwLTPhase2Task)
            self.process.rerunMvaIsolationSequence += cms.Sequence(self.process.rerunIsolationMVADBnewDMwLTPhase2Task)

            tauIDSources.byIsolationMVADBnewDMwLTPhase2raw = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "raw")
            tauIDSources.byVVLooseIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_WPEff95")
            tauIDSources.byVLooseIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_WPEff90")
            tauIDSources.byLooseIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_WPEff80")
            tauIDSources.byMediumIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_WPEff70")
            tauIDSources.byTightIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_WPEff60")
            tauIDSources.byVTightIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_WPEff50")
            tauIDSources.byVVTightIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs("rerunDiscriminationByIsolationMVADBnewDMwLTPhase2", "_WPEff40")
        ##
        if self.debug: print('Embedding new TauIDs into \"'+self.updatedTauName+'\"')
        if not hasattr(self.process, self.updatedTauName):
            embedID = cms.EDProducer("PATTauIDEmbedder",
               src = cms.InputTag('slimmedTaus'),
               tauIDSources = tauIDSources
            )
            setattr(self.process, self.updatedTauName, embedID)
        else: #assume same type
            tauIDSources = cms.PSet(
                getattr(self.process, self.updatedTauName).tauIDSources,
                tauIDSources)
            getattr(self.process, self.updatedTauName).tauIDSources = tauIDSources


    def processDeepProducer(self, producer_name, tauIDSources, workingPoints_):
        for target,points in six.iteritems(workingPoints_):
            setattr(tauIDSources, 'by{}VS{}raw'.format(producer_name[0].upper()+producer_name[1:], target),
                        cms.PSet(inputTag = cms.InputTag(producer_name, 'VS{}'.format(target)), workingPointIndex = cms.int32(-1)))
            
            cut_expressions = []
            for index, (point,cut) in enumerate(six.iteritems(points)):
                cut_expressions.append(str(cut))

                setattr(tauIDSources, 'by{}{}VS{}'.format(point, producer_name[0].upper()+producer_name[1:], target),
                        cms.PSet(inputTag = cms.InputTag(producer_name, 'VS{}'.format(target)), workingPointIndex = cms.int32(index)))

            setattr(getattr(self.process, producer_name), 'VS{}WP'.format(target), cms.vstring(*cut_expressions))


    def getDpfTauVersion(self, file_name):
        """returns the DNN version. File name should contain a version label with data takig year (2011-2, 2015-8) and \
           version number (vX), e.g. 2017v0, in general the following format: {year}v{version}"""
        version_search = re.search('201[125678]v([0-9]+)[\._]', file_name)
        if not version_search:
            raise RuntimeError('File "{}" has an invalid name pattern, should be in the format "{year}v{version}". \
                                Unable to extract version number.'.format(file_name))
        version = version_search.group(1)
        return int(version)

    def getDeepTauVersion(self, file_name):
        """returns the DeepTau year, version, subversion. File name should contain a version label with data takig year \
        (2011-2, 2015-8), version number (vX) and subversion (pX), e.g. 2017v0p6, in general the following format: \
        {year}v{version}p{subversion}"""
        version_search = re.search('(201[125678])v([0-9]+)(p[0-9]+|)[\._]', file_name)
        if not version_search:
            raise RuntimeError('File "{}" has an invalid name pattern, should be in the format "{year}v{version}p{subversion}". \
                                Unable to extract version number.'.format(file_name))
        year = version_search.group(1)
        version = version_search.group(2)
        subversion = version_search.group(3)
        if len(subversion) > 0:
            subversion = subversion[1:]
        else:
            subversion = 0
        return int(year), int(version), int(subversion)

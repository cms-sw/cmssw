from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import patDiscriminationByIsolationMVArun2v1raw, patDiscriminationByIsolationMVArun2v1
from RecoTauTag.RecoTau.DeepTau_cfi import DeepTau
from RecoTauTag.RecoTau.DeepTauIdSonicProducer_cfi import DeepTauIdSonicProducer
from RecoTauTag.RecoTau.tauIdWPsDefs import WORKING_POINTS_v2p1, WORKING_POINTS_v2p5, WORKING_POINTS_PHASEII_v2p5

import os
import re

class TauIDEmbedder(object):
    """class to rerun the tau seq and acces trainings from the database"""
    availableDiscriminators = [
        "2017v1", "2017v2", "newDM2017v2", "dR0p32017v2", "2016v1", "newDM2016v1",
        "deepTau2017v2", "deepTau2017v2p1", "deepTau2018v2p5", "deepTau2026v2p5",
        "againstEle2018",
        "newDMPhase2v1",
        "againstElePhase2v1"
    ]

    def __init__(self, process, debug = False,
                 originalTauName = "slimmedTaus",
                 updatedTauName = "slimmedTausNewID",
                 postfix = "",
                 toKeep =  ["deepTau2017v2p1", "deepTau2018v2p5", "deepTau2026v2p5"],
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
        self.originalTauName = originalTauName
        self.updatedTauName = updatedTauName
        self.postfix = postfix
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
        _rerunMvaIsolationTask = cms.Task()
        _rerunMvaIsolationSequence = cms.Sequence()
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

            _byIsolationOldDMMVArun2017v1raw = "rerunDiscriminationByIsolationOldDMMVArun2017v1raw"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2017v1raw,patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1"),
                mvaOpt = cms.string("DBoldDMwLTwGJ"),
                verbosity = cms.int32(0)
            ))

            _byIsolationOldDMMVArun2017v1 = "rerunDiscriminationByIsolationOldDMMVArun2017v1"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2017v1,patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationOldDMMVArun2017v1raw,
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
            ))

            _rerunIsolationOldDMMVArun2017v1Task =  cms.Task(
                getattr(self.process,_byIsolationOldDMMVArun2017v1raw),
                getattr(self.process,_byIsolationOldDMMVArun2017v1)
            )
            _rerunMvaIsolationTask.add(_rerunIsolationOldDMMVArun2017v1Task)
            _rerunMvaIsolationSequence += cms.Sequence(_rerunIsolationOldDMMVArun2017v1Task)

            tauIDSources.byIsolationMVArun2017v1DBoldDMwLTraw2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v1, "raw")
            tauIDSources.byVVLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v1, "_WPEff95")
            tauIDSources.byVLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v1, "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v1, "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v1, "_WPEff70")
            tauIDSources.byTightIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v1, "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v1, "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2017v1DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v1, "_WPEff40")


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

            _byIsolationOldDMMVArun2017v2raw = "rerunDiscriminationByIsolationOldDMMVArun2017v2raw"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2017v2raw,patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2"),
                mvaOpt = cms.string("DBoldDMwLTwGJ"),
                verbosity = cms.int32(0)
            ))

            _byIsolationOldDMMVArun2017v2 = "rerunDiscriminationByIsolationOldDMMVArun2017v2"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2017v2,patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationOldDMMVArun2017v2raw,
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
            ))

            _rerunIsolationOldDMMVArun2017v2Task = cms.Task(
                getattr(self.process,_byIsolationOldDMMVArun2017v2raw),
                getattr(self.process,_byIsolationOldDMMVArun2017v2)
            )
            _rerunMvaIsolationTask.add(_rerunIsolationOldDMMVArun2017v2Task)
            _rerunMvaIsolationSequence += cms.Sequence(_rerunIsolationOldDMMVArun2017v2Task)

            tauIDSources.byIsolationMVArun2017v2DBoldDMwLTraw2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v2, "raw")
            tauIDSources.byVVLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v2, "_WPEff95")
            tauIDSources.byVLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v2, "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v2, "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v2, "_WPEff70")
            tauIDSources.byTightIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v2, "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v2, "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2017v2DBoldDMwLT2017 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2017v2, "_WPEff40")

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

            _byIsolationNewDMMVArun2017v2raw = "rerunDiscriminationByIsolationNewDMMVArun2017v2raw"+self.postfix
            setattr(self.process,_byIsolationNewDMMVArun2017v2raw,patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2"),
                mvaOpt = cms.string("DBnewDMwLTwGJ"),
                verbosity = cms.int32(0)
            ))

            _byIsolationNewDMMVArun2017v2 = "rerunDiscriminationByIsolationNewDMMVArun2017v2"+self.postfix
            setattr(self.process,_byIsolationNewDMMVArun2017v2,patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationNewDMMVArun2017v2raw,
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
            ))

            _rerunIsolationNewDMMVArun2017v2Task = cms.Task(
                getattr(self.process,_byIsolationNewDMMVArun2017v2raw),
                getattr(self.process,_byIsolationNewDMMVArun2017v2)
            )
            _rerunMvaIsolationTask.add(_rerunIsolationNewDMMVArun2017v2Task)
            _rerunMvaIsolationSequence += cms.Sequence(_rerunIsolationNewDMMVArun2017v2Task)

            tauIDSources.byIsolationMVArun2017v2DBnewDMwLTraw2017 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2017v2, "raw")
            tauIDSources.byVVLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2017v2, "_WPEff95")
            tauIDSources.byVLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2017v2, "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2017v2, "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2017v2, "_WPEff70")
            tauIDSources.byTightIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2017v2, "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2017v2, "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2017v2DBnewDMwLT2017 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2017v2, "_WPEff40")

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

            _byIsolationOldDMdR0p3MVArun2017v2raw = "rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2raw"+self.postfix
            setattr(self.process,_byIsolationOldDMdR0p3MVArun2017v2raw,patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2"),
                mvaOpt = cms.string("DBoldDMwLTwGJ"),
                srcChargedIsoPtSum = cms.string('chargedIsoPtSumdR03'),
                srcFootprintCorrection = cms.string('footprintCorrectiondR03'),
                srcNeutralIsoPtSum = cms.string('neutralIsoPtSumdR03'),
                srcPhotonPtSumOutsideSignalCone = cms.string('photonPtSumOutsideSignalConedR03'),
                verbosity = cms.int32(0)
            ))

            _byIsolationOldDMdR0p3MVArun2017v2 = "rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2"+self.postfix
            setattr(self.process,_byIsolationOldDMdR0p3MVArun2017v2,patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationOldDMdR0p3MVArun2017v2raw,
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
            ))

            _rerunIsolationOldDMdR0p3MVArun2017v2Task = cms.Task(
                getattr(self.process,_byIsolationOldDMdR0p3MVArun2017v2raw),
                getattr(self.process,_byIsolationOldDMdR0p3MVArun2017v2)
            )
            _rerunMvaIsolationTask.add(_rerunIsolationOldDMdR0p3MVArun2017v2Task)
            _rerunMvaIsolationSequence += cms.Sequence(_rerunIsolationOldDMdR0p3MVArun2017v2Task)

            tauIDSources.byIsolationMVArun2017v2DBoldDMdR0p3wLTraw2017 = self.tauIDMVAinputs(_byIsolationOldDMdR0p3MVArun2017v2, "raw")
            tauIDSources.byVVLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs(_byIsolationOldDMdR0p3MVArun2017v2, "_WPEff95")
            tauIDSources.byVLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs(_byIsolationOldDMdR0p3MVArun2017v2, "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs(_byIsolationOldDMdR0p3MVArun2017v2, "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs(_byIsolationOldDMdR0p3MVArun2017v2, "_WPEff70")
            tauIDSources.byTightIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs(_byIsolationOldDMdR0p3MVArun2017v2, "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs(_byIsolationOldDMdR0p3MVArun2017v2, "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.tauIDMVAinputs(_byIsolationOldDMdR0p3MVArun2017v2, "_WPEff40")

        # 2016 training strategy(v2) - essentially the same as 2017 training strategy (v1), trained on 2016MC, old DM - currently not implemented in the tau sequence of any release
        # self.process.rerunDiscriminationByIsolationOldDMMVArun2v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
        #     PATTauProducer = self.originalTauName,
        #     Prediscriminants = noPrediscriminants,
        #     loadMVAfromDB = cms.bool(True),
        #     mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v2"),
        #     mvaOpt = cms.string("DBoldDMwLTwGJ"),
        #     verbosity = cms.int32(0)
        # )
        # #
        # self.process.rerunDiscriminationByIsolationOldDMMVArun2v2VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
        #     PATTauProducer = self.originalTauName,
        #     Prediscriminants = noPrediscriminants,
        #     toMultiplex = 'rerunDiscriminationByIsolationOldDMMVArun2v2raw',
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
            _byIsolationOldDMMVArun2016v1raw = "rerunDiscriminationByIsolationOldDMMVArun2v1raw"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2016v1raw,patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1"),
                mvaOpt = cms.string("DBoldDMwLT"),
                verbosity = cms.int32(0)
            ))

            _byIsolationOldDMMVArun2016v1 = "rerunDiscriminationByIsolationOldDMMVArun2v1"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2016v1,patDiscriminationByIsolationMVArun2v1.clone(
                    PATTauProducer = self.originalTauName,
                    Prediscriminants = noPrediscriminants,
                    toMultiplex = _byIsolationOldDMMVArun2016v1raw,
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
                ))

            _rerunIsolationOldDMMVArun2016v1Task = cms.Task(
                getattr(self.process,_byIsolationOldDMMVArun2016v1raw),
                getattr(self.process,_byIsolationOldDMMVArun2016v1)
            )
            _rerunMvaIsolationTask.add(_rerunIsolationOldDMMVArun2016v1Task)
            _rerunMvaIsolationSequence += cms.Sequence(_rerunIsolationOldDMMVArun2016v1Task)

            tauIDSources.byIsolationMVArun2v1DBoldDMwLTraw2016 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2016v1, "raw")
            tauIDSources.byVLooseIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2016v1, "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2016v1, "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2016v1, "_WPEff70")
            tauIDSources.byTightIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2016v1, "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2016v1, "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2v1DBoldDMwLT2016 = self.tauIDMVAinputs(_byIsolationOldDMMVArun2016v1, "_WPEff40")

        # 2016 training strategy(v1), trained on 2016MC, new DM
        if "newDM2016v1" in self.toKeep:
            _byIsolationNewDMMVArun2016v1raw = "rerunDiscriminationByIsolationNewDMMVArun2v1raw"+self.postfix
            setattr(self.process,_byIsolationNewDMMVArun2016v1raw,patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1"),
                mvaOpt = cms.string("DBnewDMwLT"),
                verbosity = cms.int32(0)
            ))

            _byIsolationNewDMMVArun2016v1 = "rerunDiscriminationByIsolationNewDMMVArun2v1"+self.postfix
            setattr(self.process,_byIsolationNewDMMVArun2016v1,patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationNewDMMVArun2016v1raw,
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
            ))

            _rerunIsolationNewDMMVArun2016v1Task = cms.Task(
                getattr(self.process,_byIsolationNewDMMVArun2016v1raw),
                getattr(self.process,_byIsolationNewDMMVArun2016v1)
            )
            _rerunMvaIsolationTask.add(_rerunIsolationNewDMMVArun2016v1Task)
            _rerunMvaIsolationSequence += cms.Sequence(_rerunIsolationNewDMMVArun2016v1Task)

            tauIDSources.byIsolationMVArun2v1DBnewDMwLTraw2016 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2016v1, "raw")
            tauIDSources.byVLooseIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2016v1, "_WPEff90")
            tauIDSources.byLooseIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2016v1, "_WPEff80")
            tauIDSources.byMediumIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2016v1, "_WPEff70")
            tauIDSources.byTightIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2016v1, "_WPEff60")
            tauIDSources.byVTightIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2016v1, "_WPEff50")
            tauIDSources.byVVTightIsolationMVArun2v1DBnewDMwLT2016 = self.tauIDMVAinputs(_byIsolationNewDMMVArun2016v1, "_WPEff40")

        if "deepTau2017v2" in self.toKeep:
            if self.debug: print ("Adding DeepTau v2 IDs")

            _deepTauName = "deepTau2017v2"
            workingPoints_ = WORKING_POINTS_v2p1

            file_names = [
                'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.pb',
                'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_inner.pb',
                'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_outer.pb',
            ]
            full_version = self.getDeepTauVersion(file_names[0])
            setattr(self.process,_deepTauName+self.postfix,DeepTau.clone(
                Prediscriminants = noPrediscriminants,
                taus             = self.originalTauName,
                graph_file       = file_names,
                year             = full_version[0],
                version          = full_version[1],
                sub_version      = 1 #MB: subversion cannot be properly deduced from file names; it should be 1 also for v2
            ))

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)

            from Configuration.ProcessModifiers.deepTauSonicTriton_cff import deepTauSonicTriton
            deepTauSonicTriton.toReplaceWith(_deepTauProducer, DeepTauIdSonicProducer.clone(
                Client = cms.PSet(
                  mode = cms.string('PseudoAsync'),
                  allowedTries = cms.untracked.uint32(0),
                  verbose = cms.untracked.bool(False),
                  modelName = cms.string("deeptau_2017v2p1"),
                  modelVersion = cms.string(''),
                  modelConfigPath = cms.FileInPath("RecoTauTag/TrainingFiles/data/DeepTauIdSONIC/deeptau_2017v2p1/config.pbtxt"),
                  preferredServer = cms.untracked.string(''),
                  timeout = cms.untracked.uint32(300),
                  useSharedMemory = cms.untracked.bool(True),
                  compression = cms.untracked.string(''),
                  outputs = cms.untracked.vstring()
                ),
                Prediscriminants = noPrediscriminants,
                taus             = self.originalTauName,
                year             = full_version[0],
                version          = full_version[1],
                sub_version      = 1, #MB: subversion cannot be properly deduced from file names
            ))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer


        if "deepTau2017v2p1" in self.toKeep:
            if self.debug: print ("Adding DeepTau v2p1 IDs")

            _deepTauName = "deepTau2017v2p1"
            workingPoints_ = WORKING_POINTS_v2p1

            file_names = [
                'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.pb',
                'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_inner.pb',
                'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_outer.pb',
            ]
            full_version = self.getDeepTauVersion(file_names[0])
            setattr(self.process,_deepTauName+self.postfix,DeepTau.clone(
                Prediscriminants = noPrediscriminants,
                taus             = self.originalTauName,
                graph_file       = file_names,
                year             = full_version[0],
                version          = full_version[1],
                sub_version      = 1, #MB: subversion cannot be properly deduced from file names
                disable_dxy_pca  = True
            ))

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)

            from Configuration.ProcessModifiers.deepTauSonicTriton_cff import deepTauSonicTriton
            deepTauSonicTriton.toReplaceWith(_deepTauProducer, DeepTauIdSonicProducer.clone(
                Client = cms.PSet(
                  mode = cms.string('PseudoAsync'),
                  allowedTries = cms.untracked.uint32(0),
                  verbose = cms.untracked.bool(False),
                  modelName = cms.string("deeptau_2017v2p1"),
                  modelVersion = cms.string(''),
                  modelConfigPath = cms.FileInPath("RecoTauTag/TrainingFiles/data/DeepTauIdSONIC/deeptau_2017v2p1/config.pbtxt"),
                  preferredServer = cms.untracked.string(''),
                  timeout = cms.untracked.uint32(300),
                  useSharedMemory = cms.untracked.bool(True),
                  compression = cms.untracked.string(''),
                  outputs = cms.untracked.vstring()
                ),
                Prediscriminants = noPrediscriminants,
                taus             = self.originalTauName,
                year             = full_version[0],
                version          = full_version[1],
                sub_version      = 1, #MB: subversion cannot be properly deduced from file names
                disable_dxy_pca  = True
            ))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer

        if "deepTau2018v2p5" in self.toKeep:
            if self.debug: print ("Adding DeepTau v2p5 IDs")

            _deepTauName = "deepTau2018v2p5"
            workingPoints_ = WORKING_POINTS_v2p5

            file_names = [
                'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2018v2p5_core.pb',
                'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2018v2p5_inner.pb',
                'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2018v2p5_outer.pb',
            ]
            full_version = self.getDeepTauVersion(file_names[0])
            setattr(self.process,_deepTauName+self.postfix,DeepTau.clone(
                Prediscriminants                = noPrediscriminants,
                taus                            = self.originalTauName,
                graph_file                      = file_names,
                year                            = full_version[0],
                version                         = full_version[1],
                sub_version                     = full_version[2],
                disable_dxy_pca                 = True,
                disable_hcalFraction_workaround = True,
                disable_CellIndex_workaround    = True
            ))

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)

            from Configuration.ProcessModifiers.deepTauSonicTriton_cff import deepTauSonicTriton
            deepTauSonicTriton.toReplaceWith(_deepTauProducer, DeepTauIdSonicProducer.clone(
                Client = cms.PSet(
                  mode = cms.string('PseudoAsync'),
                  allowedTries = cms.untracked.uint32(0),
                  verbose = cms.untracked.bool(False),
                  modelName = cms.string("deeptau_2018v2p5"),
                  modelVersion = cms.string(''),
                  modelConfigPath = cms.FileInPath("RecoTauTag/TrainingFiles/data/DeepTauIdSONIC/deeptau_2018v2p5/config.pbtxt"),
                  preferredServer = cms.untracked.string(''),
                  timeout = cms.untracked.uint32(300),
                  useSharedMemory = cms.untracked.bool(True),
                  compression = cms.untracked.string(''),
                  outputs = cms.untracked.vstring(),
                ),
                Prediscriminants                = noPrediscriminants,
                taus                            = self.originalTauName,
                year                            = full_version[0],
                version                         = full_version[1],
                sub_version                     = full_version[2],
                disable_dxy_pca                 = True,
                disable_hcalFraction_workaround = True,
                disable_CellIndex_workaround    = True
            ))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer

        if "deepTau2026v2p5" in self.toKeep:
            if self.debug: print ("Adding Phase2 DeepTau v2p5 IDs")

            _deepTauName = "deepTau2026v2p5"
            workingPoints_ = WORKING_POINTS_PHASEII_v2p5

            file_names = [
                'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2026v2p5_core.pb',
                'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2026v2p5_inner.pb',
                'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2026v2p5_outer.pb',
            ]
            full_version = self.getDeepTauVersion(file_names[0])
            setattr(self.process,_deepTauName+self.postfix,DeepTau.clone(
                Prediscriminants                = noPrediscriminants,
                taus                            = self.originalTauName,
                graph_file                      = file_names,
                year                            = full_version[0],
                version                         = full_version[1],
                sub_version                     = full_version[2],
                disable_dxy_pca                 = True,
                disable_hcalFraction_workaround = True,
                disable_CellIndex_workaround    = True
            ))

            from RecoTauTag.RecoTau.mergedPhase2SlimmedElectronsForTauId_cff import mergedSlimmedElectronsForTauId
            if not hasattr(self.process,"mergedSlimmedElectronsForTauId"):
                self.process.mergedSlimmedElectronsForTauId = mergedSlimmedElectronsForTauId
            setattr(getattr(self.process, _deepTauName+self.postfix), "electrons", cms.InputTag("mergedSlimmedElectronsForTauId"))
            setattr(getattr(self.process, _deepTauName+self.postfix), "vertices", cms.InputTag("offlineSlimmedPrimaryVertices4D"))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)
            _rerunMvaIsolationTask.add(self.process.mergedSlimmedElectronsForTauId)
            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += self.process.mergedSlimmedElectronsForTauId
            _rerunMvaIsolationSequence += _deepTauProducer

        if "againstEle2018" in self.toKeep:
            antiElectronDiscrMVA6_version = "MVA6v3_noeveto"
            ### Define new anti-e discriminants
            ## Raw
            from RecoTauTag.RecoTau.patTauDiscriminationAgainstElectronMVA6_cfi import patTauDiscriminationAgainstElectronMVA6
            _byElectronRejectionMVA62018Raw = "patTauDiscriminationByElectronRejectionMVA62018Raw"+self.postfix
            setattr(self.process,_byElectronRejectionMVA62018Raw,patTauDiscriminationAgainstElectronMVA6.clone(
                PATTauProducer = self.originalTauName,
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
            ))
            ## WPs
            from RecoTauTag.RecoTau.PATTauDiscriminantCutMultiplexer_cfi import patTauDiscriminantCutMultiplexer
            _byElectronRejectionMVA62018 = "patTauDiscriminationByElectronRejectionMVA62018"+self.postfix
            setattr(self.process,"patTauDiscriminationByElectronRejectionMVA62018"+self.postfix,patTauDiscriminantCutMultiplexer.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byElectronRejectionMVA62018Raw,
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
            ))
            ### Put all new anti-e discrminats to a sequence
            _patTauDiscriminationByElectronRejectionMVA62018Task = cms.Task(
                getattr(self.process,_byElectronRejectionMVA62018Raw),
                getattr(self.process,_byElectronRejectionMVA62018)
            )
            _rerunMvaIsolationTask.add(_patTauDiscriminationByElectronRejectionMVA62018Task)
            _rerunMvaIsolationSequence += cms.Sequence(_patTauDiscriminationByElectronRejectionMVA62018Task)

            _againstElectronTauIDSources = cms.PSet(
                againstElectronMVA6Raw2018 = self.tauIDMVAinputs(_byElectronRejectionMVA62018, "raw"),
                againstElectronMVA6category2018 = self.tauIDMVAinputs(_byElectronRejectionMVA62018, "category"),
                againstElectronVLooseMVA62018 = self.tauIDMVAinputs(_byElectronRejectionMVA62018, "_WPeff98"),
                againstElectronLooseMVA62018 = self.tauIDMVAinputs(_byElectronRejectionMVA62018, "_WPeff90"),
                againstElectronMediumMVA62018 = self.tauIDMVAinputs(_byElectronRejectionMVA62018, "_WPeff80"),
                againstElectronTightMVA62018 = self.tauIDMVAinputs(_byElectronRejectionMVA62018, "_WPeff70"),
                againstElectronVTightMVA62018 = self.tauIDMVAinputs(_byElectronRejectionMVA62018, "_WPeff60")
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
            _byIsolationNewDMMVAPhase2raw = "rerunDiscriminationByIsolationMVADBnewDMwLTPhase2raw"+self.postfix
            setattr(self.process,_byIsolationNewDMMVAPhase2raw,patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = True,
                mvaName = 'RecoTauTag_tauIdMVAIsoPhase2',
                mvaOpt = 'DBnewDMwLTwGJPhase2',
                verbosity = 0
            ))

            _byIsolationNewDMMVAPhase2 = "rerunDiscriminationByIsolationMVADBnewDMwLTPhase2"+self.postfix
            setattr(self.process,_byIsolationNewDMMVAPhase2,patDiscriminationByIsolationMVArun2v1.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationNewDMMVAPhase2raw,
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
                    "_VVLoose",
                    "_VLoose",
                    "_Loose",
                    "_Medium",
                    "_Tight",
                    "_VTight",
                    "_VVTight"
                )
            ))
            _rerunIsolationMVADBnewDMwLTPhase2Task = cms.Task(
                getattr(self.process,_byIsolationNewDMMVAPhase2raw),
                getattr(self.process,_byIsolationNewDMMVAPhase2)
            )
            _rerunMvaIsolationTask.add(_rerunIsolationMVADBnewDMwLTPhase2Task)
            _rerunMvaIsolationSequence += cms.Sequence(_rerunIsolationMVADBnewDMwLTPhase2Task)

            tauIDSources.byIsolationMVADBnewDMwLTPhase2raw = tauIDMVAinputs(_byIsolationNewDMMVAPhase2, "raw")
            tauIDSources.byVVLooseIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs(_byIsolationNewDMMVAPhase2, "_VVLoose")
            tauIDSources.byVLooseIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs(_byIsolationNewDMMVAPhase2, "_VLoose")
            tauIDSources.byLooseIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs(_byIsolationNewDMMVAPhase2, "_Loose")
            tauIDSources.byMediumIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs(_byIsolationNewDMMVAPhase2, "_Medium")
            tauIDSources.byTightIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs(_byIsolationNewDMMVAPhase2, "_Tight")
            tauIDSources.byVTightIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs(_byIsolationNewDMMVAPhase2, "_VTight")
            tauIDSources.byVVTightIsolationMVADBnewDMwLTPhase2 = tauIDMVAinputs(_byIsolationNewDMMVAPhase2, "_VVTight")

        if "againstElePhase2v1" in self.toKeep:
            if self.debug: print ("Adding anti-e Phase2v1 ID")
            ### Define new anti-e discriminants for Phase2
            ## Raw
            from RecoTauTag.RecoTau.PATTauDiscriminationAgainstElectronMVA6Phase2_cff import patTauDiscriminationAgainstElectronMVA6Phase2Raw, patTauDiscriminationAgainstElectronMVA6Phase2, mergedSlimmedElectronsForTauId
            _byElectronRejectionMVA6Phase2v1Raw = "patTauDiscriminationByElectronRejectionMVA6Phase2v1Raw"+self.postfix
            setattr(self.process,_byElectronRejectionMVA6Phase2v1Raw,patTauDiscriminationAgainstElectronMVA6Phase2Raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants #already selected for MiniAOD
            ))
            ## WPs
            _byElectronRejectionMVA6Phase2v1 = "patTauDiscriminationByElectronRejectionMVA6Phase2v1"+self.postfix
            setattr(self.process,_byElectronRejectionMVA6Phase2v1,patTauDiscriminationAgainstElectronMVA6Phase2.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byElectronRejectionMVA6Phase2v1Raw
            ))
            ### Put all new phase2 anti-e discrminats to a sequence
            if not hasattr(self.process,"mergedSlimmedElectronsForTauId"):
                self.process.mergedSlimmedElectronsForTauId = mergedSlimmedElectronsForTauId
            _patTauDiscriminationByElectronRejectionMVA6Phase2v1Task = cms.Task(
                self.process.mergedSlimmedElectronsForTauId,
                getattr(self.process,_byElectronRejectionMVA6Phase2v1Raw),
                getattr(self.process,_byElectronRejectionMVA6Phase2v1)
            )
            _rerunMvaIsolationTask.add(_patTauDiscriminationByElectronRejectionMVA6Phase2v1Task)
            _rerunMvaIsolationSequence += cms.Sequence(_patTauDiscriminationByElectronRejectionMVA6Phase2v1Task)

            _againstElectronTauIDPhase2v1Sources = cms.PSet(
                againstElectronMVA6RawPhase2v1 = self.tauIDMVAinputs(_byElectronRejectionMVA6Phase2v1, "raw"),
                againstElectronMVA6categoryPhase2v1 = self.tauIDMVAinputs(_byElectronRejectionMVA6Phase2v1, "category"),
                againstElectronVLooseMVA6Phase2v1 = self.tauIDMVAinputs(_byElectronRejectionMVA6Phase2v1, "_VLoose"),
                againstElectronLooseMVA6Phase2v1 = self.tauIDMVAinputs(_byElectronRejectionMVA6Phase2v1, "_Loose"),
                againstElectronMediumMVA6Phase2v1 = self.tauIDMVAinputs(_byElectronRejectionMVA6Phase2v1, "_Medium"),
                againstElectronTightMVA6Phase2v1 = self.tauIDMVAinputs(_byElectronRejectionMVA6Phase2v1, "_Tight"),
                againstElectronVTightMVA6Phase2v1 = self.tauIDMVAinputs(_byElectronRejectionMVA6Phase2v1, "_VTight")
            )
            _tauIDSourcesWithAgainistElePhase2v1 = cms.PSet(
                tauIDSources.clone(),
                _againstElectronTauIDPhase2v1Sources
            )
            tauIDSources =_tauIDSourcesWithAgainistElePhase2v1.clone()
        ##
        if self.debug: print('Embedding new TauIDs into \"'+self.updatedTauName+'\"')
        if not hasattr(self.process, self.updatedTauName):
            embedID = cms.EDProducer("PATTauIDEmbedder",
               src = cms.InputTag(self.originalTauName),
               tauIDSources = tauIDSources
            )
            setattr(self.process, self.updatedTauName, embedID)
        else: #assume same type
            tauIDSources = cms.PSet(
                getattr(self.process, self.updatedTauName).tauIDSources,
                tauIDSources)
            getattr(self.process, self.updatedTauName).tauIDSources = tauIDSources
        if not hasattr(self.process,"rerunMvaIsolationTask"+self.postfix):
            setattr(self.process,"rerunMvaIsolationTask"+self.postfix,_rerunMvaIsolationTask)
        else:
            _updatedRerunMvaIsolationTask = getattr(self.process,"rerunMvaIsolationTask"+self.postfix)
            _updatedRerunMvaIsolationTask.add(_rerunMvaIsolationTask)
            setattr(self.process,"rerunMvaIsolationTask"+self.postfix,_updatedRerunMvaIsolationTask)
        if not hasattr(self.process,"rerunMvaIsolationSequence"+self.postfix):
            setattr(self.process,"rerunMvaIsolationSequence"+self.postfix,_rerunMvaIsolationSequence)
        else:
            _updatedRerunMvaIsolationSequence = getattr(self.process,"rerunMvaIsolationSequence"+self.postfix)
            _updatedRerunMvaIsolationSequence += _rerunMvaIsolationSequence
            setattr(self.process,"rerunMvaIsolationSequence"+self.postfix,_updatedRerunMvaIsolationSequence)


    def processDeepProducer(self, producer_name, tauIDSources, workingPoints_):
        for target,points in workingPoints_.items():
            setattr(tauIDSources, 'by{}VS{}raw'.format(producer_name[0].upper()+producer_name[1:], target),
                        cms.PSet(inputTag = cms.InputTag(producer_name+self.postfix, 'VS{}'.format(target)), workingPointIndex = cms.int32(-1)))
            
            cut_expressions = []
            for index, (point,cut) in enumerate(points.items()):
                cut_expressions.append(str(cut))

                setattr(tauIDSources, 'by{}{}VS{}'.format(point, producer_name[0].upper()+producer_name[1:], target),
                        cms.PSet(inputTag = cms.InputTag(producer_name+self.postfix, 'VS{}'.format(target)), workingPointIndex = cms.int32(index)))
            if len(cut_expressions) > 0:
                setattr(getattr(self.process, producer_name+self.postfix), 'VS{}WP'.format(target), cms.vstring(*cut_expressions))


    def getDeepTauVersion(self, file_name):
        """returns the DeepTau year, version, subversion. File name should contain a version label with data takig year \
        (2011-2, 2015-8), version number (vX) and subversion (pX), e.g. 2017v0p6, in general the following format: \
        {year}v{version}p{subversion}"""
        version_search = re.search('(20[1,2][125678])v([0-9]+)(p[0-9]+|)[\._]', file_name)
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

from __future__ import print_function
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import patDiscriminationByIsolationMVArun2v1raw, patDiscriminationByIsolationMVArun2v1VLoose
import os
import re
import six

class TauIDEmbedder(object):
    """class to rerun the tau seq and acces trainings from the database"""
    availableDiscriminators = [
        "2017v1", "2017v2", "newDM2017v2", "dR0p32017v2", "2016v1", "newDM2016v1",
        "deepTau2017v1", "deepTau2017v2", "deepTau2017v2p1",
        "DPFTau_2016_v0", "DPFTau_2016_v1",
        "againstEle2018"
    ]

    def __init__(self, process, cms, debug = False,
                 originalTauName = "slimmedTaus",
                 updatedTauName = "slimmedTausNewID",
                 postfix = "",
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
        self.cms = cms
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
        if v[2] == "X":
            patch = -1
        else:
            patch = int(v[2])
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

    def loadMVA_WPs_run2_2017(self):
        if self.debug: print ("loadMVA_WPs_run2_2017: performed")
        global cms
        for training, gbrForestName in self.tauIdDiscrMVA_trainings_run2_2017.items():

            self.process.loadRecoTauTagMVAsFromPrepDB.toGet.append(
                self.cms.PSet(
                    record = self.cms.string('GBRWrapperRcd'),
                    tag = self.cms.string("RecoTauTag_%s%s" % (gbrForestName, self.tauIdDiscrMVA_2017_version)),
                    label = self.cms.untracked.string("RecoTauTag_%s%s" % (gbrForestName, self.tauIdDiscrMVA_2017_version))
                )
            )

            for WP in self.tauIdDiscrMVA_WPs_run2_2017[training].keys():
                self.process.loadRecoTauTagMVAsFromPrepDB.toGet.append(
                    self.cms.PSet(
                        record = self.cms.string('PhysicsTGraphPayloadRcd'),
                        tag = self.cms.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, self.tauIdDiscrMVA_2017_version, WP)),
                        label = self.cms.untracked.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, self.tauIdDiscrMVA_2017_version, WP))
                    )
                )

            self.process.loadRecoTauTagMVAsFromPrepDB.toGet.append(
                self.cms.PSet(
                    record = self.cms.string('PhysicsTFormulaPayloadRcd'),
                    tag = self.cms.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, self.tauIdDiscrMVA_2017_version)),
                    label = self.cms.untracked.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, self.tauIdDiscrMVA_2017_version))
                )
            )

    def runTauID(self):
        _rerunMvaIsolationTask = self.cms.Task()
        _rerunMvaIsolationSequence = self.cms.Sequence()
        tauIDSources = self.cms.PSet()

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
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1"),
                mvaOpt = self.cms.string("DBoldDMwLTwGJ"),
                verbosity = self.cms.int32(0)
            ))

            _byIsolationOldDMMVArun2017v1VLoose = "rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2017v1VLoose,patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationOldDMMVArun2017v1raw,
                key = _byIsolationOldDMMVArun2017v1raw+':category',
                loadMVAfromDB = self.cms.bool(True),
                mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
                mapping = self.cms.VPSet(
                    self.cms.PSet(
                        category = self.cms.uint32(0),
                        cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff90"), #writeTauIdDiscrWPs
                        variable = self.cms.string("pt"),
                    )
                )
            ))
            _byIsolationOldDMMVArun2017v1VLooseProd = getattr(self.process,_byIsolationOldDMMVArun2017v1VLoose)

            _rerunIsolationOldDMMVArun2017v1Task = self.cms.Task(
                getattr(self.process,_byIsolationOldDMMVArun2017v1raw),
                _byIsolationOldDMMVArun2017v1VLooseProd
            )
            tauIDSources.byIsolationMVArun2017v1DBoldDMwLTraw2017 = self.cms.InputTag(_byIsolationOldDMMVArun2017v1raw)
            tauIDSources.byVLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.cms.InputTag(_byIsolationOldDMMVArun2017v1VLoose)

            wp_to_eff_match = {
                "VVLoose" : "WPEff95",
                "Loose" : "WPEff80",
                "Medium" : "WPEff70",
                "Tight" : "WPEff60",
                "VTight" : "WPEff50",
                "VVTight" : "WPEff40"
            }
            for wp,wpEff in wp_to_eff_match.items():
                _aWP = "rerunDiscriminationByIsolationOldDMMVArun2017v1"+wp+self.postfix
                setattr(self.process,_aWP,_byIsolationOldDMMVArun2017v1VLooseProd.clone())
                _aWPProd = getattr(self.process,_aWP)
                _aWPProd.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_"+wpEff)
                _rerunIsolationOldDMMVArun2017v1Task.add(_aWPProd)
                setattr(tauIDSources,"by"+wp+"IsolationMVArun2017v1DBoldDMwLT2017",self.cms.InputTag(_aWP))

            _rerunMvaIsolationTask.add(_rerunIsolationOldDMMVArun2017v1Task)
            _rerunMvaIsolationSequence += self.cms.Sequence(_rerunIsolationOldDMMVArun2017v1Task)

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
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2"),
                mvaOpt = self.cms.string("DBoldDMwLTwGJ"),
                verbosity = self.cms.int32(0)
            ))

            _byIsolationOldDMMVArun2017v2VLoose = "rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2017v2VLoose,patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationOldDMMVArun2017v2raw,
                key = _byIsolationOldDMMVArun2017v2raw+':category',
                loadMVAfromDB = self.cms.bool(True),
                mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
                mapping = self.cms.VPSet(
                    self.cms.PSet(
                        category = self.cms.uint32(0),
                        cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff90"), #writeTauIdDiscrWPs
                        variable = self.cms.string("pt"),
                    )
                ),
                verbosity = self.cms.int32(0)
            ))
            _byIsolationOldDMMVArun2017v2VLooseProd = getattr(self.process,_byIsolationOldDMMVArun2017v2VLoose)

            _rerunIsolationOldDMMVArun2017v2Task = self.cms.Task(
                getattr(self.process,_byIsolationOldDMMVArun2017v2raw),
                _byIsolationOldDMMVArun2017v2VLooseProd
            )
            tauIDSources.byIsolationMVArun2017v2DBoldDMwLTraw2017 = self.cms.InputTag(_byIsolationOldDMMVArun2017v2raw)
            tauIDSources.byVLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.cms.InputTag(_byIsolationOldDMMVArun2017v2VLoose)

            wp_to_eff_match = {
                "VVLoose" : "WPEff95",
                "Loose" : "WPEff80",
                "Medium" : "WPEff70",
                "Tight" : "WPEff60",
                "VTight" : "WPEff50",
                "VVTight" : "WPEff40"
            }
            for wp,wpEff in wp_to_eff_match.items():
                _aWP = "rerunDiscriminationByIsolationOldDMMVArun2017v2"+wp+self.postfix
                setattr(self.process,_aWP,_byIsolationOldDMMVArun2017v2VLooseProd.clone())
                _aWPProd = getattr(self.process,_aWP)
                _aWPProd.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_"+wpEff)
                _rerunIsolationOldDMMVArun2017v2Task.add(_aWPProd)
                setattr(tauIDSources,"by"+wp+"IsolationMVArun2017v2DBoldDMwLT2017",self.cms.InputTag(_aWP))

            _rerunMvaIsolationTask.add(_rerunIsolationOldDMMVArun2017v2Task)
            _rerunMvaIsolationSequence += self.cms.Sequence(_rerunIsolationOldDMMVArun2017v2Task)

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
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2"),
                mvaOpt = self.cms.string("DBnewDMwLTwGJ"),
                verbosity = self.cms.int32(0)
            ))

            _byIsolationNewDMMVArun2017v2VLoose = "rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose"+self.postfix
            setattr(self.process,_byIsolationNewDMMVArun2017v2VLoose,patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationNewDMMVArun2017v2raw,
                key = _byIsolationNewDMMVArun2017v2raw+':category',
                loadMVAfromDB = self.cms.bool(True),
                mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
                mapping = self.cms.VPSet(
                    self.cms.PSet(
                        category = self.cms.uint32(0),
                        cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff90"), #writeTauIdDiscrWPs
                        variable = self.cms.string("pt"),
                    )
                ),
                verbosity = self.cms.int32(0)
            ))
            _byIsolationNewDMMVArun2017v2VLooseProd = getattr(self.process,_byIsolationNewDMMVArun2017v2VLoose)

            _rerunIsolationNewDMMVArun2017v2Task = self.cms.Task(
                getattr(self.process,_byIsolationNewDMMVArun2017v2raw),
                _byIsolationNewDMMVArun2017v2VLooseProd
            )
            tauIDSources.byIsolationMVArun2017v2DBnewDMwLTraw2017 = self.cms.InputTag(_byIsolationNewDMMVArun2017v2raw)
            tauIDSources.byVLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.cms.InputTag(_byIsolationNewDMMVArun2017v2VLoose)

            wp_to_eff_match = {
                "VVLoose" : "WPEff95",
                "Loose" : "WPEff80",
                "Medium" : "WPEff70",
                "Tight" : "WPEff60",
                "VTight" : "WPEff50",
                "VVTight" : "WPEff40"
            }
            for wp,wpEff in wp_to_eff_match.items():
                _aWP = "rerunDiscriminationByIsolationNewDMMVArun2017v2"+wp+self.postfix
                setattr(self.process,_aWP,_byIsolationNewDMMVArun2017v2VLooseProd.clone())
                _aWPProd = getattr(self.process,_aWP)
                _aWPProd.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_"+wpEff)
                _rerunIsolationNewDMMVArun2017v2Task.add(_aWPProd)
                setattr(tauIDSources,"by"+wp+"IsolationMVArun2017v2DBnewDMwLT2017",self.cms.InputTag(_aWP))

            _rerunMvaIsolationTask.add(_rerunIsolationNewDMMVArun2017v2Task)
            _rerunMvaIsolationSequence += self.cms.Sequence(_rerunIsolationNewDMMVArun2017v2Task)

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
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2"),
                mvaOpt = self.cms.string("DBoldDMwLTwGJ"),
                srcChargedIsoPtSum = self.cms.string('chargedIsoPtSumdR03'),
                srcFootprintCorrection = self.cms.string('footprintCorrectiondR03'),
                srcNeutralIsoPtSum = self.cms.string('neutralIsoPtSumdR03'),
                srcPhotonPtSumOutsideSignalCone = self.cms.string('photonPtSumOutsideSignalConedR03'),
                verbosity = self.cms.int32(0)
            ))

            _byIsolationOldDMdR0p3MVArun2017v2VLoose = "rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose"+self.postfix
            setattr(self.process,_byIsolationOldDMdR0p3MVArun2017v2VLoose,patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationOldDMdR0p3MVArun2017v2raw,
                key = _byIsolationOldDMdR0p3MVArun2017v2raw+':category',
                loadMVAfromDB = self.cms.bool(True),
                mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
                mapping = self.cms.VPSet(
                    self.cms.PSet(
                        category = self.cms.uint32(0),
                        cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff90"), #writeTauIdDiscrWPs
                        variable = self.cms.string("pt"),
                    )
                ),
                verbosity = self.cms.int32(0)
            ))
            _byIsolationOldDMdR0p3MVArun2017v2VLooseProd = getattr(self.process,_byIsolationOldDMdR0p3MVArun2017v2VLoose)

            _rerunIsolationOldDMdR0p3MVArun2017v2Task = self.cms.Task(
                getattr(self.process,_byIsolationOldDMdR0p3MVArun2017v2raw),
                _byIsolationOldDMdR0p3MVArun2017v2VLooseProd
            )
            tauIDSources.byIsolationMVArun2017v2DBoldDMdR0p3wLTraw2017 = self.cms.InputTag(_byIsolationOldDMdR0p3MVArun2017v2raw)
            tauIDSources.byVLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.cms.InputTag(_byIsolationOldDMdR0p3MVArun2017v2VLoose)

            wp_to_eff_match = {
                "VVLoose" : "WPEff95",
                "Loose" : "WPEff80",
                "Medium" : "WPEff70",
                "Tight" : "WPEff60",
                "VTight" : "WPEff50",
                "VVTight" : "WPEff40"
            }
            for wp,wpEff in wp_to_eff_match.items():
                _aWP = "rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2"+wp+self.postfix
                setattr(self.process,_aWP,_byIsolationOldDMdR0p3MVArun2017v2VLooseProd.clone())
                _aWPProd = getattr(self.process,_aWP)
                _aWPProd.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_"+wpEff)
                _rerunIsolationOldDMdR0p3MVArun2017v2Task.add(_aWPProd)
                setattr(tauIDSources,"by"+wp+"IsolationMVArun2017v2DBoldDMdR0p3wLT2017",self.cms.InputTag(_aWP))

            _rerunMvaIsolationTask.add(_rerunIsolationOldDMdR0p3MVArun2017v2Task)
            _rerunMvaIsolationSequence += self.cms.Sequence(_rerunIsolationOldDMdR0p3MVArun2017v2Task)

        # 2016 training strategy(v2) - essentially the same as 2017 training strategy (v1), trained on 2016MC, old DM - currently not implemented in the tau sequence of any release
        # self.process.rerunDiscriminationByIsolationOldDMMVArun2v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
        #     PATTauProducer = self.originalTauName,
        #     Prediscriminants = noPrediscriminants,
        #     loadMVAfromDB = self.cms.bool(True),
        #     mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v2"),
        #     mvaOpt = self.cms.string("DBoldDMwLTwGJ"),
        #     verbosity = self.cms.int32(0)
        # )
        # #
        # self.process.rerunDiscriminationByIsolationOldDMMVArun2v2VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
        #     PATTauProducer = self.originalTauName,
        #     Prediscriminants = noPrediscriminants,
        #     toMultiplex = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v2raw'),
        #     key = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v2raw:category'),#?
        #     loadMVAfromDB = self.cms.bool(True),
        #     mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v2_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
        #     mapping = self.cms.VPSet(
        #         self.cms.PSet(
        #             category = self.cms.uint32(0),
        #             cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v2_WPEff90"), #writeTauIdDiscrWPs
        #             variable = self.cms.string("pt"),
        #         )
        #     )
        # )

        # 2016 training strategy(v1), trained on 2016MC, old DM
        if "2016v1" in self.toKeep:
            _byIsolationOldDMMVArun2016v1raw = "rerunDiscriminationByIsolationOldDMMVArun2v1raw"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2016v1raw,patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1"),
                mvaOpt = self.cms.string("DBoldDMwLT"),
                verbosity = self.cms.int32(0)
            ))

            _byIsolationOldDMMVArun2016v1VLoose = "rerunDiscriminationByIsolationOldDMMVArun2v1VLoose"+self.postfix
            setattr(self.process,_byIsolationOldDMMVArun2016v1VLoose,patDiscriminationByIsolationMVArun2v1VLoose.clone(
                    PATTauProducer = self.originalTauName,
                    Prediscriminants = noPrediscriminants,
                    toMultiplex = _byIsolationOldDMMVArun2016v1raw,
                    key = _byIsolationOldDMMVArun2016v1raw+':category',
                    loadMVAfromDB = self.cms.bool(True),
                    mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_mvaOutput_normalization"),
                    mapping = self.cms.VPSet(
                        self.cms.PSet(
                            category = self.cms.uint32(0),
                            cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff90"),
                            variable = self.cms.string("pt"),
                        )
                    )
                ))
            _byIsolationOldDMMVArun2016v1VLooseProd = getattr(self.process,_byIsolationOldDMMVArun2016v1VLoose)

            _rerunIsolationOldDMMVArun2016v1Task = self.cms.Task(
                getattr(self.process,_byIsolationOldDMMVArun2016v1raw),
                _byIsolationOldDMMVArun2016v1VLooseProd
            )
            tauIDSources.byIsolationMVArun2v1DBoldDMwLTraw2016 = self.cms.InputTag(_byIsolationOldDMMVArun2016v1raw)
            tauIDSources.byVLooseIsolationMVArun2v2DBoldDMwLT2016 = self.cms.InputTag(_byIsolationOldDMMVArun2016v1VLoose)

            wp_to_eff_match = {
                "Loose" : "WPEff80",
                "Medium" : "WPEff70",
                "Tight" : "WPEff60",
                "VTight" : "WPEff50",
                "VVTight" : "WPEff40"
            }
            for wp,wpEff in wp_to_eff_match.items():
                _aWP = "rerunDiscriminationByIsolationOldDMMVArun2v1"+wp+self.postfix
                setattr(self.process,_aWP,_byIsolationOldDMMVArun2016v1VLooseProd.clone())
                _aWPProd = getattr(self.process,_aWP)
                _aWPProd.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_"+wpEff)
                _rerunIsolationOldDMMVArun2016v1Task.add(_aWPProd)
                setattr(tauIDSources,"by"+wp+"IsolationMVArun2v1DBoldDMwLT2016",self.cms.InputTag(_aWP))

            _rerunMvaIsolationTask.add(_rerunIsolationOldDMMVArun2016v1Task)
            _rerunMvaIsolationSequence += self.cms.Sequence(_rerunIsolationOldDMMVArun2016v1Task)

        # 2016 training strategy(v1), trained on 2016MC, new DM
        if "newDM2016v1" in self.toKeep:
            _byIsolationNewDMMVArun2016v1raw = "rerunDiscriminationByIsolationNewDMMVArun2v1raw"+self.postfix
            setattr(self.process,_byIsolationNewDMMVArun2016v1raw,patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1"),
                mvaOpt = self.cms.string("DBnewDMwLT"),
                verbosity = self.cms.int32(0)
            ))

            _byIsolationNewDMMVArun2016v1VLoose = "rerunDiscriminationByIsolationNewDMMVArun2v1VLoose"+self.postfix
            setattr(self.process,_byIsolationNewDMMVArun2016v1VLoose,patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byIsolationNewDMMVArun2016v1raw,
                key = _byIsolationNewDMMVArun2016v1raw+':category',
                loadMVAfromDB = self.cms.bool(True),
                mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_mvaOutput_normalization"),
                mapping = self.cms.VPSet(
                    self.cms.PSet(
                        category = self.cms.uint32(0),
                        cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_WPEff90"),
                        variable = self.cms.string("pt"),
                    )
                )
            ))
            _byIsolationNewDMMVArun2016v1VLooseProd = getattr(self.process,_byIsolationNewDMMVArun2016v1VLoose)

            _rerunIsolationNewDMMVArun2016v1Task = self.cms.Task(
                getattr(self.process,_byIsolationNewDMMVArun2016v1raw),
                _byIsolationNewDMMVArun2016v1VLooseProd
            )
            tauIDSources.byIsolationMVArun2v1DBnewDMwLTraw2016 = self.cms.InputTag(_byIsolationNewDMMVArun2016v1raw)
            tauIDSources.byVLooseIsolationMVArun2v2DBnewDMwLT2016 = self.cms.InputTag(_byIsolationNewDMMVArun2016v1VLoose)

            wp_to_eff_match = {
                "Loose" : "WPEff80",
                "Medium" : "WPEff70",
                "Tight" : "WPEff60",
                "VTight" : "WPEff50",
                "VVTight" : "WPEff40"
            }
            for wp,wpEff in wp_to_eff_match.items():
                _aWP = "rerunDiscriminationByIsolationNewDMMVArun2v1"+wp+self.postfix
                setattr(self.process,_aWP,_byIsolationNewDMMVArun2016v1VLooseProd.clone())
                _aWPProd = getattr(self.process,_aWP)
                _aWPProd.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_"+wpEff)
                _rerunIsolationNewDMMVArun2016v1Task.add(_aWPProd)
                setattr(tauIDSources,"by"+wp+"IsolationMVArun2v1DBnewDMwLT2016",self.cms.InputTag(_aWP))

            _rerunMvaIsolationTask.add(_rerunIsolationNewDMMVArun2016v1Task)
            _rerunMvaIsolationSequence += self.cms.Sequence(_rerunIsolationNewDMMVArun2016v1Task)

        if "deepTau2017v1" in self.toKeep:
            if self.debug: print ("Adding DeepTau IDs")

            _deepTauName = "deepTau2017v1"
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
            setattr(self.process,_deepTauName+self.postfix,self.cms.EDProducer("DeepTauId",
                electrons              = self.cms.InputTag('slimmedElectrons'),
                muons                  = self.cms.InputTag('slimmedMuons'),
                taus                   = self.cms.InputTag(self.originalTauName),
                pfcands                = self.cms.InputTag('packedPFCandidates'),
                vertices               = self.cms.InputTag('offlineSlimmedPrimaryVertices'),
                rho                    = self.cms.InputTag('fixedGridRhoAll'),
                graph_file             = self.cms.vstring(file_names),
                mem_mapped             = self.cms.bool(False),
                version                = self.cms.uint32(self.getDeepTauVersion(file_names[0])[1]),
                debug_level            = self.cms.int32(0),
                disable_dxy_pca        = self.cms.bool(False)

            ))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)
            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer

        if "deepTau2017v2" in self.toKeep:
            if self.debug: print ("Adding DeepTau IDs")

            _deepTauName = "deepTau2017v2"
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
            setattr(self.process,_deepTauName+self.postfix,self.cms.EDProducer("DeepTauId",
                electrons              = self.cms.InputTag('slimmedElectrons'),
                muons                  = self.cms.InputTag('slimmedMuons'),
                taus                   = self.cms.InputTag(self.originalTauName),
                pfcands                = self.cms.InputTag('packedPFCandidates'),
                vertices               = self.cms.InputTag('offlineSlimmedPrimaryVertices'),
                rho                    = self.cms.InputTag('fixedGridRhoAll'),
                graph_file             = self.cms.vstring(file_names),
                mem_mapped             = self.cms.bool(True),
                version                = self.cms.uint32(self.getDeepTauVersion(file_names[0])[1]),
                debug_level            = self.cms.int32(0),
                disable_dxy_pca        = self.cms.bool(False)

            ))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)
            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer

        if "deepTau2017v2p1" in self.toKeep:
            if self.debug: print ("Adding DeepTau IDs")

            _deepTauName = "deepTau2017v2p1"
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
            setattr(self.process,_deepTauName+self.postfix,self.cms.EDProducer("DeepTauId",
                electrons                = self.cms.InputTag('slimmedElectrons'),
                muons                    = self.cms.InputTag('slimmedMuons'),
                taus                     = self.cms.InputTag(self.originalTauName),
                pfcands                  = self.cms.InputTag('packedPFCandidates'),
                vertices                 = self.cms.InputTag('offlineSlimmedPrimaryVertices'),
                rho                      = self.cms.InputTag('fixedGridRhoAll'),
                graph_file               = self.cms.vstring(file_names),
                mem_mapped               = self.cms.bool(True),
                version                  = self.cms.uint32(self.getDeepTauVersion(file_names[0])[1]),
                debug_level              = self.cms.int32(0),
                disable_dxy_pca          = self.cms.bool(True)

            ))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)
            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer

        if "DPFTau_2016_v0" in self.toKeep:
            if self.debug: print ("Adding DPFTau isolation (v0)")

            _deepTauName = "DPFTau_2016_v0"
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
            setattr(self.process,_deepTauName+self.postfix,self.cms.EDProducer("DPFIsolation",
                pfcands     = self.cms.InputTag('packedPFCandidates'),
                taus        = self.cms.InputTag(self.originalTauName),
                vertices    = self.cms.InputTag('offlineSlimmedPrimaryVertices'),
                graph_file  = self.cms.vstring(file_names),
                version     = self.cms.uint32(self.getDpfTauVersion(file_names[0])),
                mem_mapped  = self.cms.bool(False)
            ))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)
            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer

        if "DPFTau_2016_v1" in self.toKeep:
            print ("Adding DPFTau isolation (v1)")
            print ("WARNING: WPs are not defined for DPFTau_2016_v1")
            print ("WARNING: The score of DPFTau_2016_v1 is inverted: i.e. for Sig->0, for Bkg->1 with -1 for undefined input (preselection not passed).")

            _deepTauName = "DPFTau_2016_v1"
            workingPoints_ = {
                "all": {"Tight" : 0.123} #FIXME: define WP
            }

            file_names = [ 'RecoTauTag/TrainingFiles/data/DPFTauId/DPFIsolation_2017v1_quantized.pb' ]
            setattr(self.process,_deepTauName+self.postfix,self.cms.EDProducer("DPFIsolation",
                pfcands     = self.cms.InputTag('packedPFCandidates'),
                taus        = self.cms.InputTag(self.originalTauName),
                vertices    = self.cms.InputTag('offlineSlimmedPrimaryVertices'),
                graph_file  = self.cms.vstring(file_names),
                version     = self.cms.uint32(self.getDpfTauVersion(file_names[0])),
                mem_mapped  = self.cms.bool(False)
            ))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)
            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer

        if "againstEle2018" in self.toKeep:
            antiElectronDiscrMVA6_version = "MVA6v3_noeveto"
            ### Define new anti-e discriminants
            ## Raw
            from RecoTauTag.RecoTau.PATTauDiscriminationAgainstElectronMVA6_cfi import patTauDiscriminationAgainstElectronMVA6
            _byElectronRejectionMVA62018Raw = "patTauDiscriminationByElectronRejectionMVA62018Raw"+self.postfix
            setattr(self.process,_byElectronRejectionMVA62018Raw,patTauDiscriminationAgainstElectronMVA6.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants, #already selected for MiniAOD
                vetoEcalCracks = self.cms.bool(False), #keep taus in EB-EE cracks
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
            # VLoose
            _byElectronRejectionMVA62018VLoose = "patTauDiscriminationByVLooseElectronRejectionMVA62018"+self.postfix
            setattr(self.process,_byElectronRejectionMVA62018VLoose,patTauDiscriminantCutMultiplexer.clone(
                PATTauProducer = self.originalTauName,
                Prediscriminants = noPrediscriminants,
                toMultiplex = _byElectronRejectionMVA62018Raw,
                key = _byElectronRejectionMVA62018Raw+":category",
                mapping = self.cms.VPSet(
                    self.cms.PSet(
                        category = self.cms.uint32(0),
                        cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_woGwoGSF_BL_WPeff98'),
                        variable = self.cms.string('pt')
                    ),
                    self.cms.PSet(
                        category = self.cms.uint32(2),
                        cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_wGwoGSF_BL_WPeff98'),
                        variable = self.cms.string('pt')
                    ),
                    self.cms.PSet(
                        category = self.cms.uint32(5),
                        cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_woGwGSF_BL_WPeff98'),
                        variable = self.cms.string('pt')
                    ),
                    self.cms.PSet(
                        category = self.cms.uint32(7),
                        cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_wGwGSF_BL_WPeff98'),
                        variable = self.cms.string('pt')
                    ),
                    self.cms.PSet(
                        category = self.cms.uint32(8),
                        cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_woGwoGSF_EC_WPeff98'),
                        variable = self.cms.string('pt')
                    ),
                    self.cms.PSet(
                        category = self.cms.uint32(10),
                        cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_wGwoGSF_EC_WPeff98'),
                        variable = self.cms.string('pt')
                    ),
                    self.cms.PSet(
                        category = self.cms.uint32(13),
                        cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_woGwGSF_EC_WPeff98'),
                        variable = self.cms.string('pt')
                    ),
                    self.cms.PSet(
                        category = self.cms.uint32(15),
                        cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_wGwGSF_EC_WPeff98'),
                        variable = self.cms.string('pt')
                    )
                )
            ))
            _byElectronRejectionMVA62018VLooseProd = getattr(self.process,_byElectronRejectionMVA62018VLoose)

            _patTauDiscriminationByElectronRejectionMVA62018Task = self.cms.Task(
                getattr(self.process,_byElectronRejectionMVA62018Raw),
                _byElectronRejectionMVA62018VLooseProd
            )
            _againstElectronTauIDSources = self.cms.PSet(
                againstElectronMVA6Raw2018 = self.cms.InputTag(_byElectronRejectionMVA62018Raw),
                againstElectronMVA6category2018 = self.cms.InputTag(_byElectronRejectionMVA62018Raw,"category"),
                againstElectronVLooseMVA62018 = self.cms.InputTag(_byElectronRejectionMVA62018VLoose)
            )
            # Other WPs
            wp_to_eff_match = {
                "Loose" : "WPeff90",
                "Medium" : "WPeff80",
                "Tight" : "WPeff70",
                "VTight" : "WPeff60",
            }
            for wp,wpEff in wp_to_eff_match.items():
                _aWP = "patTauDiscriminationBy"+wp+"ElectronRejectionMVA62018"+self.postfix
                setattr(self.process,_aWP,_byElectronRejectionMVA62018VLooseProd.clone(
                    mapping = self.cms.VPSet(
                        self.cms.PSet(
                            category = self.cms.uint32(0),
                            cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_woGwoGSF_BL_'+wpEff),
                            variable = self.cms.string('pt')
                        ),
                        self.cms.PSet(
                            category = self.cms.uint32(2),
                            cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_wGwoGSF_BL_'+wpEff),
                            variable = self.cms.string('pt')
                        ),
                        self.cms.PSet(
                            category = self.cms.uint32(5),
                            cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_woGwGSF_BL_'+wpEff),
                            variable = self.cms.string('pt')
                        ),
                        self.cms.PSet(
                            category = self.cms.uint32(7),
                            cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_wGwGSF_BL_'+wpEff),
                            variable = self.cms.string('pt')
                        ),
                        self.cms.PSet(
                            category = self.cms.uint32(8),
                            cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_woGwoGSF_EC_'+wpEff),
                            variable = self.cms.string('pt')
                        ),
                        self.cms.PSet(
                            category = self.cms.uint32(10),
                            cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_NoEleMatch_wGwoGSF_EC_'+wpEff),
                            variable = self.cms.string('pt')
                        ),
                        self.cms.PSet(
                            category = self.cms.uint32(13),
                            cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_woGwGSF_EC_'+wpEff),
                            variable = self.cms.string('pt')
                        ),
                        self.cms.PSet(
                            category = self.cms.uint32(15),
                            cut = self.cms.string('RecoTauTag_antiElectron'+antiElectronDiscrMVA6_version+'_gbr_wGwGSF_EC_'+wpEff),
                            variable = self.cms.string('pt')
                        )
                    )
                ))
                _patTauDiscriminationByElectronRejectionMVA62018Task.add(getattr(self.process,_aWP))
                setattr(_againstElectronTauIDSources,"againstElectron"+wp+"MVA62018",self.cms.InputTag(_aWP))

            _rerunMvaIsolationTask.add(_patTauDiscriminationByElectronRejectionMVA62018Task)
            _rerunMvaIsolationSequence += self.cms.Sequence(_patTauDiscriminationByElectronRejectionMVA62018Task)

            _tauIDSourcesWithAgainistEle = self.cms.PSet(
                tauIDSources.clone(),
                _againstElectronTauIDSources
            )
            tauIDSources =_tauIDSourcesWithAgainistEle.clone()

        ##
        if self.debug: print('Embedding new TauIDs into \"'+self.updatedTauName+'\"')
        if not hasattr(self.process, self.updatedTauName):
            embedID = self.cms.EDProducer("PATTauIDEmbedder",
               src = self.cms.InputTag(self.originalTauName),
               tauIDSources = tauIDSources
            )
            setattr(self.process, self.updatedTauName, embedID)
        else: #assume same type
            tauIDSources = self.cms.PSet(
                getattr(self.process, self.updatedTauName).tauIDSources,
                tauIDSources)
            getattr(self.process, self.updatedTauName).tauIDSources = tauIDSources
        setattr(self.process,"rerunMvaIsolationTask"+self.postfix,_rerunMvaIsolationTask)
        setattr(self.process,"rerunMvaIsolationSequence"+self.postfix,_rerunMvaIsolationSequence)


    def processDeepProducer(self, producer_name, tauIDSources, workingPoints_):
        for target,points in six.iteritems(workingPoints_):
            cuts = self.cms.PSet()
            setattr(tauIDSources, 'by{}VS{}raw'.format(producer_name[0].upper()+producer_name[1:], target),
                        self.cms.InputTag(producer_name+self.postfix, 'VS{}'.format(target)))
            for point,cut in six.iteritems(points):
                setattr(cuts, point, self.cms.string(str(cut)))

                setattr(tauIDSources, 'by{}{}VS{}'.format(point, producer_name[0].upper()+producer_name[1:], target),
                        self.cms.InputTag(producer_name+self.postfix, 'VS{}{}'.format(target, point)))

            setattr(getattr(self.process, producer_name+self.postfix), 'VS{}WP'.format(target), cuts)


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

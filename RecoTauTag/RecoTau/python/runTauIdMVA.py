from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import patDiscriminationByIsolationMVArun2v1raw, patDiscriminationByIsolationMVArun2v1VLoose
import os

class TauIDEmbedder(object):
    """class to rerun the tau seq and acces trainings from the database"""

    def __init__(self, process, cms, debug = False,
        toKeep = ["2016v1", "newDM2016v1","deepTau2017v1","DPFTau_2016_v0"],
        tauIdDiscrMVA_trainings_run2_2017 = {
            'tauIdMVAIsoDBoldDMwLT2017' : "tauIdMVAIsoDBoldDMwLT2017",
        },
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
        self.toKeep = toKeep


    @staticmethod
    def get_cmssw_version(debug = False):
        """returns 'CMSSW_X_Y_Z'"""
        if debug: print "get_cmssw_version:", os.environ["CMSSW_RELEASE_BASE"].split('/')[-1]
        return os.environ["CMSSW_RELEASE_BASE"].split('/')[-1]

    @classmethod
    def get_cmssw_version_number(klass, debug = False):
        """returns 'X_Y_Z' (without 'CMSSW_')"""
        if debug: print "get_cmssw_version_number:", map(int, klass.get_cmssw_version().split("CMSSW_")[1].split("_")[0:3])
        return map(int, klass.get_cmssw_version().split("CMSSW_")[1].split("_")[0:3])

    @staticmethod
    def versionToInt(release=9, subversion=4, patch=0, debug = False):
        if debug: print "versionToInt:", release * 10000 + subversion * 100 + patch
        return release * 10000 + subversion * 100 + patch

    @classmethod
    def is_above_cmssw_version(klass, release=9, subversion=4, patch=0, debug = False):
        split_cmssw_version = klass.get_cmssw_version_number()
        if klass.versionToInt(release, subversion, patch) > klass.versionToInt(split_cmssw_version[0], split_cmssw_version[1], split_cmssw_version[2]):
            if debug: print "is_above_cmssw_version:", False
            return False
        else:
            if debug: print "is_above_cmssw_version:", True
            return True

    def loadMVA_WPs_run2_2017(self):
        if self.debug: print "loadMVA_WPs_run2_2017: performed"
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

    def runTauID(self, name='NewTauIDsEmbedded'):
        self.process.rerunMvaIsolationSequence = self.cms.Sequence()
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
            if not self.is_above_cmssw_version(10, 0, 0, self.debug):
                if self.debug: print "runTauID: not is_above_cmssw_version(10, 0, 0). Will update the list of available in DB samples to access 2017v1"
                self.loadMVA_WPs_run2_2017()

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1"),#RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1 writeTauIdDiscrMVAs
                mvaOpt = self.cms.string("DBoldDMwLTwGJ"),
                requireDecayMode = self.cms.bool(True),
                verbosity = self.cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1raw'),
                key = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1raw:category'),#?
                loadMVAfromDB = self.cms.bool(True),
                mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_mvaOutput_normalization"), #writeTauIdDiscrMVAoutputNormalizations
                mapping = self.cms.VPSet(
                    self.cms.PSet(
                        category = self.cms.uint32(0),
                        cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff90"), #writeTauIdDiscrWPs
                        variable = self.cms.string("pt"),
                    )
                )
            )

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VVLoose = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VVLoose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff95")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1Loose = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1Loose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff80")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1Medium = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1Medium.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff70")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1Tight = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1Tight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff60")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VTight = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff50")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VVTight = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VVTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v1_WPEff40")

            self.process.rerunMvaIsolationSequence += self.cms.Sequence(
                self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1raw
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VVLoose
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1Loose
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1Medium
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1Tight
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VTight
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v1VVTight
            )

            tauIDSources.byIsolationMVArun2017v1DBoldDMwLTraw2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1raw')
            tauIDSources.byVVLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1VVLoose')
            tauIDSources.byVLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1VLoose')
            tauIDSources.byLooseIsolationMVArun2017v1DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1Loose')
            tauIDSources.byMediumIsolationMVArun2017v1DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1Medium')
            tauIDSources.byTightIsolationMVArun2017v1DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1Tight')
            tauIDSources.byVTightIsolationMVArun2017v1DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1VTight')
            tauIDSources.byVVTightIsolationMVArun2017v1DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v1VVTight')


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

            if self.debug: print "runTauID: not is_above_cmssw_version(10, 0, 0). Will update the list of available in DB samples to access 2017v2"
            self.loadMVA_WPs_run2_2017()

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2"),#RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1 writeTauIdDiscrMVAs
                mvaOpt = self.cms.string("DBoldDMwLTwGJ"),
                requireDecayMode = self.cms.bool(True),
                verbosity = self.cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2raw'),
                key = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2raw:category'),#?
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
            )

            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VVLoose = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VVLoose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff95")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2Loose = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2Loose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff80")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2Medium = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2Medium.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff70")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2Tight = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2Tight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff60")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VTight = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff50")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VVTight = self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VVTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff40")

            self.process.rerunMvaIsolationSequence += self.cms.Sequence(
                self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2raw
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VVLoose
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2Loose
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2Medium
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2Tight
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VTight
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2017v2VVTight
            )

            tauIDSources.byIsolationMVArun2017v2DBoldDMwLTraw2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2raw')
            tauIDSources.byVVLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2VVLoose')
            tauIDSources.byVLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2VLoose')
            tauIDSources.byLooseIsolationMVArun2017v2DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2Loose')
            tauIDSources.byMediumIsolationMVArun2017v2DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2Medium')
            tauIDSources.byTightIsolationMVArun2017v2DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2Tight')
            tauIDSources.byVTightIsolationMVArun2017v2DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2VTight')
            tauIDSources.byVVTightIsolationMVArun2017v2DBoldDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2017v2VVTight')

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

            if self.debug: print "runTauID: not is_above_cmssw_version(10, 0, 0). Will update the list of available in DB samples to access newDM2017v2"
            self.loadMVA_WPs_run2_2017()

            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2"),#RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1 writeTauIdDiscrMVAs
                mvaOpt = self.cms.string("DBnewDMwLTwGJ"),
                requireDecayMode = self.cms.bool(True),
                verbosity = self.cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2raw'),
                key = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2raw:category'),#?
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
            )

            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VVLoose = self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VVLoose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff95")
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2Loose = self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2Loose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff80")
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2Medium = self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2Medium.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff70")
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2Tight = self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2Tight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff60")
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VTight = self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff50")
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VVTight = self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VVTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff40")

            self.process.rerunMvaIsolationSequence += self.cms.Sequence(
                self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2raw
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VVLoose
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2Loose
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2Medium
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2Tight
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VTight
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2017v2VVTight
            )

            tauIDSources.byIsolationMVArun2017v2DBnewDMwLTraw2017 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2raw')
            tauIDSources.byVVLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2VVLoose')
            tauIDSources.byVLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2VLoose')
            tauIDSources.byLooseIsolationMVArun2017v2DBnewDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2Loose')
            tauIDSources.byMediumIsolationMVArun2017v2DBnewDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2Medium')
            tauIDSources.byTightIsolationMVArun2017v2DBnewDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2Tight')
            tauIDSources.byVTightIsolationMVArun2017v2DBnewDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2VTight')
            tauIDSources.byVVTightIsolationMVArun2017v2DBnewDMwLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2017v2VVTight')

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

            if self.debug: print "runTauID: not is_above_cmssw_version(10, 0, 0). Will update the list of available in DB samples to access dR0p32017v2"
            self.loadMVA_WPs_run2_2017()

            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2"),
                mvaOpt = self.cms.string("DBoldDMwLTwGJ"),
                requireDecayMode = self.cms.bool(True),
                srcChargedIsoPtSum = self.cms.string('chargedIsoPtSumdR03'),
                srcFootprintCorrection = self.cms.string('footprintCorrectiondR03'),
                srcNeutralIsoPtSum = self.cms.string('neutralIsoPtSumdR03'),
                srcPhotonPtSumOutsideSignalCone = self.cms.string('photonPtSumOutsideSignalConedR03'),
                verbosity = self.cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2raw'),
                key = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2raw:category'),#?
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
            )

            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VVLoose = self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VVLoose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff95")
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Loose = self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Loose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff80")
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Medium = self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Medium.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff70")
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Tight = self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Tight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff60")
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VTight = self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff50")
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VVTight = self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VVTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff40")

            self.process.rerunMvaIsolationSequence += self.cms.Sequence(
                self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2raw
                *self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose
                *self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VVLoose
                *self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Loose
                *self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Medium
                *self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Tight
                *self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VTight
                *self.process.rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VVTight
            )

            tauIDSources.byIsolationMVArun2017v2DBoldDMdR0p3wLTraw2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2raw')
            tauIDSources.byVVLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VVLoose')
            tauIDSources.byVLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VLoose')
            tauIDSources.byLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Loose')
            tauIDSources.byMediumIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Medium')
            tauIDSources.byTightIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2Tight')
            tauIDSources.byVTightIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VTight')
            tauIDSources.byVVTightIsolationMVArun2017v2DBoldDMdR0p3wLT2017 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMdR0p3MVArun2017v2VVTight')

        # 2016 training strategy(v2) - essentially the same as 2017 training strategy (v1), trained on 2016MC, old DM - currently not implemented in the tau sequence of any release
        # self.process.rerunDiscriminationByIsolationOldDMMVArun2v2raw = patDiscriminationByIsolationMVArun2v1raw.clone(
        #     PATTauProducer = self.cms.InputTag('slimmedTaus'),
        #     Prediscriminants = noPrediscriminants,
        #     loadMVAfromDB = self.cms.bool(True),
        #     mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v2"),#RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1 writeTauIdDiscrMVAs
        #     mvaOpt = self.cms.string("DBoldDMwLTwGJ"),
        #     requireDecayMode = self.cms.bool(True),
        #     verbosity = self.cms.int32(0)
        # )
        # #
        # self.process.rerunDiscriminationByIsolationOldDMMVArun2v2VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
        #     PATTauProducer = self.cms.InputTag('slimmedTaus'),
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
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1"),
                mvaOpt = self.cms.string("DBoldDMwLT"),
                requireDecayMode = self.cms.bool(True),
                verbosity = self.cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
                    PATTauProducer = self.cms.InputTag('slimmedTaus'),
                    Prediscriminants = noPrediscriminants,
                    toMultiplex = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1raw'),
                    key = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1raw:category'),
                    loadMVAfromDB = self.cms.bool(True),
                    mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_mvaOutput_normalization"),
                    mapping = self.cms.VPSet(
                        self.cms.PSet(
                            category = self.cms.uint32(0),
                            cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff90"),
                            variable = self.cms.string("pt"),
                        )
                    )
                )

            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1Loose = self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1Loose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff80")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1Medium = self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1Medium.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff70")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1Tight = self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1Tight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff60")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VTight = self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff50")
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VVTight = self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VVTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff40")

            self.process.rerunMvaIsolationSequence += self.cms.Sequence(
                self.process.rerunDiscriminationByIsolationOldDMMVArun2v1raw
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VLoose
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2v1Loose
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2v1Medium
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2v1Tight
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VTight
                *self.process.rerunDiscriminationByIsolationOldDMMVArun2v1VVTight
            )

            tauIDSources.byIsolationMVArun2v1DBoldDMwLTraw2016 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1raw')
            tauIDSources.byVLooseIsolationMVArun2v1DBoldDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1VLoose')
            tauIDSources.byLooseIsolationMVArun2v1DBoldDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1Loose')
            tauIDSources.byMediumIsolationMVArun2v1DBoldDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1Medium')
            tauIDSources.byTightIsolationMVArun2v1DBoldDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1Tight')
            tauIDSources.byVTightIsolationMVArun2v1DBoldDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1VTight')
            tauIDSources.byVVTightIsolationMVArun2v1DBoldDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationOldDMMVArun2v1VVTight')

        # 2016 training strategy(v1), trained on 2016MC, new DM
        if "newDM2016v1" in self.toKeep:
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1raw = patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = self.cms.bool(True),
                mvaName = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1"),
                mvaOpt = self.cms.string("DBnewDMwLT"),
                requireDecayMode = self.cms.bool(True),
                verbosity = self.cms.int32(0)
            )

            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
                PATTauProducer = self.cms.InputTag('slimmedTaus'),
                Prediscriminants = noPrediscriminants,
                toMultiplex = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1raw'),
                key = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1raw:category'),
                loadMVAfromDB = self.cms.bool(True),
                mvaOutput_normalization = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_mvaOutput_normalization"),
                mapping = self.cms.VPSet(
                    self.cms.PSet(
                        category = self.cms.uint32(0),
                        cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_WPEff90"),
                        variable = self.cms.string("pt"),
                    )
                )
            )

            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1Loose = self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1Loose.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_WPEff80")
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1Medium = self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1Medium.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_WPEff70")
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1Tight = self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1Tight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_WPEff60")
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VTight = self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_WPEff50")
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VVTight = self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VLoose.clone()
            self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VVTight.mapping[0].cut = self.cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2016v1_WPEff40")

            self.process.rerunMvaIsolationSequence += self.cms.Sequence(
                self.process.rerunDiscriminationByIsolationNewDMMVArun2v1raw
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VLoose
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2v1Loose
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2v1Medium
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2v1Tight
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VTight
                *self.process.rerunDiscriminationByIsolationNewDMMVArun2v1VVTight
            )

            tauIDSources.byIsolationMVArun2v1DBnewDMwLTraw2016 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1raw')
            tauIDSources.byVLooseIsolationMVArun2v1DBnewDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1VLoose')
            tauIDSources.byLooseIsolationMVArun2v1DBnewDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1Loose')
            tauIDSources.byMediumIsolationMVArun2v1DBnewDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1Medium')
            tauIDSources.byTightIsolationMVArun2v1DBnewDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1Tight')
            tauIDSources.byVTightIsolationMVArun2v1DBnewDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1VTight')
            tauIDSources.byVVTightIsolationMVArun2v1DBnewDMwLT2016 = self.cms.InputTag('rerunDiscriminationByIsolationNewDMMVArun2v1VVTight')

        if "deepTau2017v1" in self.toKeep:
            print "Adding DeepTau isolation?"

            from RecoTauTag.RecoTau.DeepTauId_cff import deepTauIdraw

            self.process.deepTauIdraw = deepTauIdraw.clone(
                electrons = self.cms.InputTag('slimmedElectrons'),
                muons = self.cms.InputTag('slimmedMuons'),
                taus = self.cms.InputTag('slimmedTaus'),
                graph_file = self.cms.string('RecoTauTag/RecoTau/data/deepTau_2017v1_20L1024N.pb')
            )

            self.process.rerunMvaIsolationSequence += self.cms.Sequence(
                self.process.deepTauIdraw
            )

            tauIDSources.deepTau2017v1tauVSe = self.cms.InputTag('deepTauIdraw', 'tauVSe')
            tauIDSources.deepTau2017v1tauVSmu = self.cms.InputTag('deepTauIdraw', 'tauVSmu')
            tauIDSources.deepTau2017v1tauVSjet = self.cms.InputTag('deepTauIdraw', 'tauVSjet')#$= self.cms.InputTag('deepTauIdrawtauVSjet')
            tauIDSources.deepTau2017v1tauVSall = self.cms.InputTag('deepTauIdraw', 'tauVSall')
	    print("Doing an embedding ")

	if "DPFTau_2016_v0" in self.toKeep:
            print "Adding DPF isolation?"

            from RecoTauTag.RecoTau.DPFIsolation_cff import DPFIsolation

            self.process.DPFIsolationv0 = DPFIsolation.clone(
                electrons = self.cms.InputTag('slimmedElectrons'),
                muons = self.cms.InputTag('slimmedMuons'),
                taus = self.cms.InputTag('slimmedTaus'),
                graph_file = self.cms.string('RecoTauTag/RecoTau/data/DPFIsolation_2017v0.pb')
            )

            self.process.rerunMvaIsolationSequence += self.cms.Sequence(
                self.process.DPFIsolationv0
            )

            tauIDSources.DPFTau_2016_v0tauVSe = self.cms.InputTag('DPFIsolationv0', 'tauVSe')
            tauIDSources.DPFTau_2016_v0tauVSmu = self.cms.InputTag('DPFIsolationv0', 'tauVSmu')
            tauIDSources.DPFTau_2016_v0tauVSjet = self.cms.InputTag('DPFIsolationv0', 'tauVSjet')
            tauIDSources.DPFTau_2016_v0tauVSall = self.cms.InputTag('DPFIsolationv0', 'tauVSall')

        if "DPFTau_2016_v1" in self.toKeep:
            print "Adding DPF isolation?"

            from RecoTauTag.RecoTau.DPFIsolation_cff import DPFIsolation

            self.process.DPFIsolationv1 = DPFIsolation.clone(
                electrons = self.cms.InputTag('slimmedElectrons'),
                muons = self.cms.InputTag('slimmedMuons'),
                taus = self.cms.InputTag('slimmedTaus'),
                graph_file = self.cms.string('RecoTauTag/RecoTau/data/DPFIsolation_2017v1.pb')
            )

            self.process.rerunMvaIsolationSequence += self.cms.Sequence(
                self.process.DPFIsolationv1
            )

            tauIDSources.DPFTau_2016_v1tauVSe = self.cms.InputTag('DPFIsolationv1', 'tauVSe')
            tauIDSources.DPFTau_2016_v1tauVSmu = self.cms.InputTag('DPFIsolationv1', 'tauVSmu')
            tauIDSources.DPFTau_2016_v1tauVSjet = self.cms.InputTag('DPFIsolationv1', 'tauVSjet')
            tauIDSources.DPFTau_2016_v1tauVSall = self.cms.InputTag('DPFIsolationv1', 'tauVSall')



        embedID = self.cms.EDProducer("PATTauIDEmbedder",
            src = self.cms.InputTag('slimmedTaus'),
            tauIDSources = tauIDSources
        )
        self.process.NewTauIDsEmbedded = embedID

import os
import pickle
import glob
import re

from CMGTools.RootTools.utils.splitFactor                   import splitFactor

from CMGTools.H2TauTau.proto.samples.phys14.higgs           import mc_higgs
from CMGTools.H2TauTau.proto.samples.phys14.ewk             import mc_ewk
from CMGTools.H2TauTau.proto.samples.phys14.diboson         import mc_diboson
from CMGTools.H2TauTau.proto.samples.phys14.qcd             import mc_qcd
from CMGTools.H2TauTau.proto.samples.phys14.triggers_tauMu  import mc_triggers as mc_triggers_mt
from CMGTools.H2TauTau.proto.samples.phys14.triggers_tauEle import mc_triggers as mc_triggers_et
from CMGTools.H2TauTau.proto.samples.phys14.triggers_tauTau import mc_triggers as mc_triggers_tt
from CMGTools.H2TauTau.proto.samples.phys14.triggers_muEle  import mc_triggers as mc_triggers_em

class httConnector(object):

    def __init__(self, tier, user, pattern, triggers,
                 production=False, splitFactor=10e4,
                 fineSplitFactor=4, cache=True, verbose=False):
        ''' '''
        if tier.startswith('%'):
            self.tier = tier
        else:
            self.tier = '%'+tier
        self.user            = user
        self.pattern         = pattern
        self.cache           = cache
        self.verbose         = verbose
        self.production      = production
        self.triggers        = triggers
        self.splitFactor     = splitFactor
        self.fineSplitFactor = fineSplitFactor
        self.homedir         = os.getenv('HOME')
        self.mc_dict         = {}
        self.MC_list         = []
        self.aliases         = aliases =  {
                                          '/GluGluToHToTauTau.*Phys14DR.*'            : 'HiggsGGH'         ,
                                          '/VBF_HToTauTau.*Phys14DR.*'                : 'HiggsVBF'         ,
                                          '/DYJetsToLL.*Phys14DR.*'                   : 'DYJets'           ,
                                          '/TTJets.*Phys14DR.*'                       : 'TTJets'           ,
                                          '/T_tW.*Phys14DR.*'                         : 'T_tW'             ,
                                          '/Tbar_tW.*Phys14DR.*'                      : 'Tbar_tW'          ,
                                          '/WZJetsTo3LNu.*Phys14DR.*'                 : 'WZJetsTo3LNu'     ,
                                          '/TTbarH.*Phys14DR.*'                       : 'HiggsTTHInclusive',
                                          '/WJetsToLNu.*Phys14DR.*'                   : 'WJets'            ,
                                          '/QCD_Pt-10to20_EMEnriched.*Phys14DR.*'     : 'QCDEM10to20'      ,
                                          '/QCD_Pt-20to30_EMEnriched.*Phys14DR.*'     : 'QCDEM20to30'      ,
                                          '/QCD_Pt-30to80_EMEnriched.*Phys14DR.*'     : 'QCDEM30to80'      ,
                                          '/QCD_Pt-80to170_EMEnriched.*Phys14DR.*'    : 'QCDEM80to170'     ,
                                          '/QCD_Pt-30to50_MuEnrichedPt5.*Phys14DR.*'  : 'QCDMu30to50'      ,
                                          '/QCD_Pt-50to80_MuEnrichedPt5.*Phys14DR.*'  : 'QCDMu50to80'      ,
                                          '/QCD_Pt-80to120_MuEnrichedPt5.*Phys14DR.*' : 'QCDMu80to120'     ,
                                         }
        self.dictionarize_()
        self.listify_()

    def dictionarize_(self):
        ''' '''
        for s in mc_higgs + mc_ewk + mc_diboson + mc_qcd:
            self.mc_dict[s.name] = s

    def listify_(self):
        ''' '''
        self.MC_list = [v for k, v in self.mc_dict.items()]
        for sam in self.MC_list:
            sam.splitFactor     = splitFactor(sam, self.splitFactor)
            sam.fineSplitFactor = self.fineSplitFactor
            if self.triggers == 'mt': sam.triggers = mc_triggers_mt
            if self.triggers == 'et': sam.triggers = mc_triggers_et
            if self.triggers == 'tt': sam.triggers = mc_triggers_tt
            if self.triggers == 'em': sam.triggers = mc_triggers_em

    def connect(self):
        '''Retrieves the relevant information
        (e.g. files location) for each component.
        To avoid multiple connections to the database
        is production == True, it checks for a cached
        pickle file containing all the info.

        FIXME! RIC: this should be done by default,
        but I make the use of the cached pickle explicit
        because the name of the parent dataset, and
        therefore the number of events and the
        computing efficiency, is not
        saved in the pickle file, and it can only be retrieved
        through a query to the database. This is
        necessary at the analysis level, but not at
        the production stage, where the only bit of info
        that's really needed is the location of the files.
        Should revisit the way the pickle file is saved
        so that ALL the relevant info is stored there.
        '''
        if self.production:
            self.connect_by_pck_()
        else:
            self.connect_by_db_()

        self.pruneSampleList_()

    def connect_by_db_(self):
        ''' '''
        from CMGTools.RootTools.utils.connect import connect
        connect(self.MC_list, self.tier, self.pattern,
                self.aliases, cache=self.cache, verbose=self.verbose)

    def connect_by_pck_(self):
        ''' '''
        from CMGTools.RootTools.utils.getFiles import getFiles

        redict_aliases = dict( zip(self.aliases.values(), self.aliases.keys()) )

        regex = re.compile(r'(?P<sample>[a-zA-Z0-9_]+[a-zA-Z])(?:[0-9]+)$')

        for alias_k, alias_v in self.mc_dict.items():
            m = regex.match(alias_k)
            if m and 'QCD' not in alias_k:
                alias_k = m.group('sample')
            if alias_k not in self.aliases.values():
                continue
            sample_pck = '*'.join(['',redict_aliases[alias_k].replace('/','').replace('.','*'),
                                   self.tier.replace('%',''),self.pattern+'.pck'])
            cached_sample = glob.glob('/'.join([self.homedir,'.cmgdataset',sample_pck]))
            single_mc_list = [alias_v]

            if len(cached_sample) == 0:
                print 'sample not cached yet, connecting to the DB'
                from CMGTools.RootTools.utils.connect import connect
                connect(single_mc_list, self.tier, self.pattern, self.aliases,
                         cache=self.cache, verbose=self.verbose)

            elif len(cached_sample) >1:
                print 'better specify which sample, many found'
                print cached_sample
                raise

            else:
                file = open(cached_sample[0])
                mycomp = pickle.load(file)
                single_mc_list[0].files = getFiles('/'.join( ['']+mycomp.lfnDir.split('/')[mycomp.lfnDir.split('/').index('CMG')+1:] ),
                                                              mycomp.user, self.pattern, useCache=self.cache)
                print 'attached files to %s' %(single_mc_list[0].name)
                print 'files %s' %('/'.join(single_mc_list[0].files[0].split('/')[:-1]+[self.pattern]))

    def pruneSampleList_(self):
        self.MC_list = [m for m in self.MC_list if m.files]

if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser()
    parser.usage = ''' To be written '''

    parser.add_option('-T', '--tier'      , dest = 'tier'      ,  help = 'Tier. Search samples on eos that end with this tier'                                            )
    parser.add_option('-U', '--user'      , dest = 'user'      ,  help = 'User. User or group that owns the samples. Default htautau_group'    , default = 'htautau_group')
    parser.add_option('-P', '--pattern'   , dest = 'pattern'   ,  help = 'Pattern. Connect only files that match this pattern. Default .*root' , default = '.*root'       )
    parser.add_option('-C', '--channel'   , dest = 'channel'   ,  help = 'Channel. Choose [mt, et, tt, em]. Default mt'                        , default = 'mt'           )
    parser.add_option('-p', '--production', dest = 'production',  help = 'Production. Check the cache first. Default False'                    , default = False          )

    (options,args) = parser.parse_args()

    my_connect = httConnector(options.tier, options.user, options.pattern, options.channel, options.production)
    my_connect.connect()

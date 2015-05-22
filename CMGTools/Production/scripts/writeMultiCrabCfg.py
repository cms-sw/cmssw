#!/usr/bin/env python

import os

if __name__ == '__main__':

    from optparse import OptionParser, OptionGroup
    
    usage = """usage: %prog [options] /Sample/Name/On/EOS

e.g.: %prog --cfg crab.cfg -t PAT_CMG_5_6_0 /DYJetsToLL_M-10To50filter_8TeV-madgraph/Summer12_DR53X-PU_S10_START53_V7A-v1/AODSIM

The script will write a file named 'multicrab.cfg' in the current working directory
    """
    
    parser = OptionParser(usage=usage)
    group = OptionGroup(parser,'writeMultiCrabCfg Options','Options related to multicrab')
    
    group.add_option("-c", "--cfg", dest="cfg", default='crab.cfg',help="The master crab cfg to use, e.g. 'crab.cfg'", metavar='FILE')
    group.add_option("-o", "--output", dest="output", default='multicrab.cfg',help="The multicrab cfg to write, e.g. 'multicrab.cfg'", metavar='FILE')
    group.add_option("-t", "--tier", dest="tier", default='',help="The data tier to use, e.g. 'PAT_CMG_5_6_0'")
    group.add_option("-u", "--user", dest="user", default=None,help="The user space to write into")
    parser.add_option_group(group)    
    (opts, datasets) = parser.parse_args()

    import ConfigParser
    config = ConfigParser.SafeConfigParser()
    #set options to be case sensitive
    config.optionxform = str

    config.add_section('MULTICRAB')
    config.set('MULTICRAB','cfg',opts.cfg)

    from CMGTools.Production.castorBaseDir import castorBaseDir
    import CMGTools.Production.eostools as castortools
    topdir = castortools.lfnToCastor(castorBaseDir(user=opts.user))

    output_dirs = []
    for d in datasets:

        #accept the user%dataset syntax, but ignore the user name for grid
        tokens = d.split('%')
        if len(tokens) == 2:
            d = tokens[1]

        safe_name = d.replace('/','_')
        if safe_name.startswith('_'):
            safe_name = safe_name[1:]
        if safe_name.endswith('_'):
            safe_name = safe_name[:-1]
        
        config.add_section(safe_name)

        directory = '%s/%s' % (topdir,d)
        if opts.tier:
            directory = os.path.join(directory,opts.tier)
        directory = directory.replace('//','/')

        config.set(safe_name,'CMSSW.datasetpath',d)
        lfn = castortools.castorToLFN(directory)
        config.set(safe_name,'USER.user_remote_dir',lfn)
        output_dirs.append(lfn)
        
        #create the directory on EOS
        if not castortools.fileExists(directory):
            castortools.createCastorDir(directory)
            castortools.chmod(directory,'775')
        if not castortools.isDirectory(directory): 
            raise Exception("Dataset directory '%s' does not exist or could not be created" % directory)
        
    config.write(file(opts.output,'wb'))

    from logger import logger
    logDir = 'Logger'
    os.mkdir(logDir)
    log = logger( logDir )
    log.logCMSSW()
    log.addFile( os.path.join( os.getcwd(), opts.cfg) )
    log.addFile( os.path.join( os.getcwd(), opts.output) )

    for d in output_dirs:
        log.stageOut(d)

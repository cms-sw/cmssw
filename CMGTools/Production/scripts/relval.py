#!/usr/bin/env python

from CMGTools.Production.relval import *
import imp

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    
    parser.usage = "relval.py <cfg.py> <relvalList.py>\nRuns a cfg on the batch, for a given set of RelVal datasets"
    parser.add_option("-n", "--negate", action="store_true",
                      dest="negate", default=False,
                      help="create jobs, but do nothing")
    parser.add_option("-t", "--tier", 
                      dest="tier",
                      help="Tier: extension you can give to specify you are doing a new production",
                      default=None)
    parser.add_option("-b", "--batch", 
                      dest="batch",
                      help="Batch command. Same as in cmsBatch.py",
                      default="bsub -q 1nh < batchScript.sh")
    
    import CMGTools.Production.castorBaseDir as castorBaseDir
    
#    parser.add_option("-c", "--castorBaseDir", 
#                      dest="castorBaseDir",
#                      help="Base castor directory. Subdirectories will be created automatically for each prod",
#                      default=castorBaseDir.defaultCastorBaseDir)
    
    (options,args) = parser.parse_args()

    if len(args)!=2:
        parser.print_help()
        sys.exit(1)

    cfgFileName = args[0]
    relvalListFileName = args[1]
    castorBaseDir = castorBaseDir.myCastorBaseDir()
    
    if not os.path.isfile( cfgFileName ):
        print 'cfg file does not exist: ', cfgFileName
        sys.exit(1)
    if not os.path.isfile( relvalListFileName ):
        print 'relval list file does not exist: ', relvalListFileName
        sys.exit(1)
    
    handle = open(relvalListFileName, 'r')
    cfo = imp.load_source("pycfg", relvalListFileName, handle)
    relvals = cfo.relvals
    handle.close()

    # from myRelvalList import relvals
    # loading cfg in the current directory.
    # sys.path.append('.')
    # from patTuple_PATandPF2PAT_RecoJets_cfg import process

    handle = open( cfgFileName, 'r')
    cfo = imp.load_source("pycfg", cfgFileName, handle)
    process = cfo.process
    handle.close()

    locals = []
    remotes = []
    myRelvals = []
    for relval in relvals.list:
        (local,remote) = processRelVal(relval, cfgFileName, process, options.negate, options.tier, options.batch)
        locals.append( local )
        remotes.append( remote ) 
        myRelvals.append( relval )

    print 
    print 'SUMMARY'
    print '-------'
    i = 0
    for relval in myRelvals:
        print ''
        print 'output of relval: ', relval, ' will appear in:'
        print '----------------------------------------------'
        print 'local  : '
        print locals[i]+'/*'
        print 'remote : '
        print remotes[i]
        i = i+1



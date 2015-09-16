#!/usr/bin/env python

from PhysicsTools.HeppyCore.utils.edmIntegrityCheck import PublishToFileSystem, IntegrityCheck
import das

import copy, os

if __name__ == '__main__':
    
    from optparse import OptionParser, OptionGroup
    
    usage = """usage: %prog [options] /Sample/Name/On/Castor

e.g.: %prog -u wreece -p -w 'PFAOD_*.root' /MultiJet/Run2011A-05Aug2011-v1/AOD/V2
    """
    das = das.DASOptionParser(usage=usage)
    group = OptionGroup(das.parser,'edmIntegrityCheck Options','Options related to checking files on CASTOR')
    
    group.add_option("-d", "--device", dest="device", default='cmst3',help="The storage device to write to, e.g. 'cmst3'")
    group.add_option("-n", "--name", dest="name", default=None,help='The name of the dataset in DAS. Will be guessed if not specified')
    group.add_option("-p", "--printout", dest="printout", default=False, action='store_true',help='Print a report to stdout')    
    group.add_option("-r", "--recursive", dest="resursive", default=False, action='store_true',help='Walk the mass storage device recursively')
    group.add_option("-u", "--user", dest="user", default=os.environ['USER'],help='The username to use when looking at mass storage devices')
    group.add_option("-w", "--wildcard", dest="wildcard", default=None,help='A UNIX style wildcard to specify which files to check')
    group.add_option("--update", dest="update", default=False, action='store_true',help='Only update the status of corrupted files')
    group.add_option("-t","--timeout", dest="timeout", default=-1, type=int, help='Set a timeout on the edmFileUtil calls')
    group.add_option("--min-run", dest="min_run", default=-1, type=int, help='When querying DBS, require runs >= than this run')
    group.add_option("--max-run", dest="max_run", default=-1, type=int, help='When querying DBS, require runs <= than this run')
    group.add_option("--max_threads", dest="max_threads", default=None,help='The maximum number of threads to use')
    das.parser.add_option_group(group)    
    (opts, datasets) = das.get_opt()

    if len(datasets)==0:
        print das.parser.print_help()
        print
        print 'need to provide a dataset in argument'

    def work(d,op):
        tokens = d.split('%')
        if len(tokens) == 2:
            op.user = tokens[0]
            d = tokens[1]
        
        check = IntegrityCheck(d,op)
        pub = PublishToFileSystem(check)

        previous = None
        if op.update:
            previous = pub.get(check.directory)

        check.test(previous = previous, timeout = op.timeout)
        if op.printout:
            check.report()
        report = check.structured()
        pub.publish(report)

        return d

    def callback(result):
        print 'Checking thread done: ',str(result)
    
    #submit the main work in a multi-threaded way
    import multiprocessing
    if opts.max_threads is not None and opts.max_threads:
        opts.max_threads = int(opts.max_threads)
    pool = multiprocessing.Pool(processes=opts.max_threads)

    for d in datasets:
        pool.apply_async(work, args=(d,copy.deepcopy(opts)),callback=callback)
    pool.close()
    pool.join()

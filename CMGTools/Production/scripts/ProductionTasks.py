#!/usr/bin/env python

from CMGTools.Production.ProductionTasks import *

if __name__ == '__main__':
    
    from optparse import OptionGroup
    def addOptionFromTask(task, name = None):
        if name is None:
            name = task.name
        usage = "Options for the ProductionTask '%s'" % name
        if task.__doc__:
            usage = task.__doc__
        g = OptionGroup(op.das.parser,name,usage)
        task.addOption(g)
        if g.option_list:
            op.das.parser.add_option_group(g)
    
    dataset = None
    user = os.getlogin()
    options = {}
    
    op = ParseOptions(dataset,user,options)
    addOptionFromTask(op,name=os.path.basename(sys.argv[0]))

    
    tasks = [CheckDatasetExists(dataset,user,options),
             FindOnCastor(dataset,user,options),
             CheckForMask(dataset,user,options),
             CheckForWrite(dataset,user,options),
             # BaseDataset(dataset,user,options),
             GenerateMask(dataset,user,options),
             CreateJobDirectory(dataset,user,options),             
             SourceCFG(dataset,user,options),
             FullCFG(dataset,user,options),
             CheckConfig(dataset,user,options),
             RunTestEvents(dataset,user,options),
             ExpandConfig(dataset,user,options),
             WriteToDatasets(dataset,user,options),
             RunCMSBatch(dataset,user,options),
             MonitorJobs(dataset,user,options),
             CheckJobStatus(dataset,user,options),
             CleanJobFiles(dataset,user,options),
             WriteJobReport(dataset,user,options)
             ]
    
    #allow the tasks to add extra options
    for t in tasks:
        addOptionFromTask(t)
                
    #get the options
    try:
        op.run({})
    except:
        #COLIN commented the print_help because it was kind of masking the exceptions...
        # op.das.parser.print_help()
        # print err
        sys.exit(1)
    
    def splitUser(dataSample,UserName):
        tokens = dataSample.split('%')
        if len(tokens) == 2:
            UserName = tokens[0]
            dataSample = tokens[1]
        return (dataSample,UserName)
    
    def splitTier(dataSample, tier):
        tokens = [t for t in tier.split('/') if t]
        
        sample = dataSample
        t = tier
        if len(tokens) > 1:
            sample = os.path.join(dataSample,*tokens[:-1])
            sample = sample.replace('//','/')
            t = tokens[-1]
        return sample, t
    
    def addRunRange(t, min_range, max_range):
        """Automatically decorate the tier name with the run range if its set"""

        result = t
        decorate = (min_range > 0 or max_range > 0)
        if decorate:
            start = 'start'
            end = 'end'
            if min_range > 0:
                start = str(min_range)
            if max_range > 0:
                end = str(max_range)
            if t:
                result = "%s_runrange_%s-%s" % (t,start,end)
            else:
                result = "runrange_%s-%s" % (start,end)
        return result
    
    #these tasks are quick and are done in the main thread (fail early...)
    simple_tasks = [CheckDatasetExists(dataset,user,options),FindOnCastor(dataset,user,options)]
    for d in op.dataset:
        for t in simple_tasks:
            t.options = copy.deepcopy(op.options)
            t.dataset, t.user = splitUser(d,op.user)
            t.dataset, t.options.tier = splitTier(t.dataset, t.options.tier)
            t.options.tier = addRunRange(t.options.tier, t.options.min_run, t.options.max_run)
            t.run({})
    
    def callback(result):
        print 'Production thread done: ',str(result)
    
    def log(output,s, tostdout = True):
        """Brain-dead utility function"""
        if tostdout:
            print s
        print >> output,s
    
    def work(dataset,op_parse,task_list):
        """Do the work for one dataset"""
        
        logfile = '%s.log' % dataset.replace('/','_')
        output = file(logfile,'w')
        
        previous = {}
        for t in task_list:

            t.options = copy.deepcopy(op_parse.options)
            t.dataset, t.user = splitUser(dataset,op_parse.user)
            t.dataset, t.options.tier = splitTier(t.dataset, t.options.tier)
            t.options.tier = addRunRange(t.options.tier, t.options.min_run, t.options.max_run)

            log(output,'%s: [%s] %s:' % (dataset,time.asctime(),t.getname()))
            if t.__doc__:
                log(output,'%s: %s' % (dataset,t.__doc__) )
            try:
                previous[t.getname()] = t.run(previous)
                log(output,'%s: \t%s' % (dataset,previous[t.getname()]),tostdout=False)
            except Exception, e:

                import traceback, StringIO
                sb = StringIO.StringIO()
                traceback.print_exc(file=sb)
                tb = sb.getvalue()
                sb.close()
                
                log(output,'%s: [%s] %s FAILED:' % (dataset,time.asctime(),t.getname()))
                log(output,"%s: Error was '%s'" % (dataset,str(e)))
                log(output,"%s: Traceback was '%s'" % (dataset,tb))

                #TODO: Perhaps some cleaning?
                break
            
        output.close()
        
        #dump the output in a python friendly format
        import pickle
        dumpfile = '%s.pkl' % dataset.replace('/','_')
        output = file(dumpfile,'wb')
        pickle.dump(previous, output)
        output.close()
        
        return logfile
    
    #submit the main work in a multi-threaded way
    import multiprocessing
    if op.options.max_threads is not None and op.options.max_threads:
        op.options.max_threads = int(op.options.max_threads)
    pool = multiprocessing.Pool(processes=op.options.max_threads)
    print op.dataset
    for d in op.dataset:
        pool.apply_async(work, args=(d,copy.deepcopy(op),copy.deepcopy(tasks)),callback=callback)
    pool.close()
    pool.join()

 
        
    

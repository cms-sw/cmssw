
import copy, datetime, inspect, fnmatch, os, re, subprocess, sys, tempfile, time
import glob
import gzip
import errno
from edmIntegrityCheck import PublishToFileSystem, IntegrityCheck
from addToDatasets import addToDatasets

import eostools as castortools
import das as Das

from dataset import Dataset
from datasetToSource import createDataset
from castorBaseDir import castorBaseDir

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise

class Task(object):
    """Base class for Task API"""
    def __init__(self, name, dataset, user, options, instance = None):
        self.name = name
        self.instance = instance
        self.dataset = dataset
        self.user = user
        self.options = options
    def getname(self):
        """The name of the object, using the instance if needed"""
        if self.instance is not None:
            return '%s_%s' % (self.name,self.instance)
        else:
            return self.name
    def addOption(self, parser):
        """A hook for adding things to the parser"""
        pass
    def run(self, input):
        """Basic API for a task. input and output are dictionaries"""
        return {}
    
class ParseOptions(Task):
    """Common options for the script __main__: used by all production tasks"""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'ParseOptions', dataset, user, options)

        usage = """%prog [options] <dataset>
        
The %prog script aims to take a list of samples and process them on the batch system. Submission
may be done serially (by setting --max_threads to 1), or in parallel (the default).

The basic flow is:

    1) Check that the sample to run on exists
    2) Generate a source CFG
    3) Run locally and check everything works with a small number of events
    4) Submit to the batch system 
    5) Wait until the jobs are finished
    6) Check the jobs ran OK and that the files are good

Example:

ProductionTasks.py -u cbern -w 'PFAOD*.root' -c -N 1 -q 8nh -t PAT_CMG_V5_10_0 --output_wildcard '*.root' --cfg PATCMG_cfg.py /QCD_Pt-1800_TuneZ2_7TeV_pythia6/Summer11-PU_S3_START42_V11-v2/AODSIM/V2

It is often useful to store the sample names in a file, in which case you could instead do:

ProductionTasks.py -w '*.root' -c -N 1 -q 8nh -t PAT_CMG_V5_10_0 --output_wildcard '*.root' --cfg PATCMG_cfg.py `cat samples_mc.txt`

An example file might contain:

palencia%/Tbar_TuneZ2_tW-channel-DR_7TeV-powheg-tauola/Summer11-PU_S4_START42_V11-v1/AODSIM/V2
benitezj%/ZZ_TuneZ2_7TeV_pythia6_tauola/Summer11-PU_S4_START42_V11-v1/AODSIM/V2
wreece%/ZJetsToNuNu_100_HT_200_7TeV-madgraph/Summer11-PU_S4_START42_V11-v1/AODSIM/V2

The CASTOR username for each sample is given before the '%'.

Each step in the flow has a task associated with it, which may set options. The options for each task are 
documented below.

"""
        self.das = Das.DASOptionParser(usage=usage)
    def addOption(self, parser):
        parser.add_option("-u", "--user", dest="user", default=os.getlogin(),help='The username to use when looking at mass storage devices. Your login username is used by default.')
        parser.add_option("-w", "--wildcard", dest="wildcard", default='*.root',help='A UNIX style wildcard to specify which input files to check before submitting the jobs')    
        parser.add_option("--max_threads", dest="max_threads", default=None,help='The maximum number of threads to use in the production')
    def run(self, input):
        self.options, self.dataset = self.das.get_opt()
        self.dataset = [d for d in self.dataset if not d.startswith('#')]
        self.user = self.options.user
        if not self.dataset:
            raise Exception('TaskError: No dataset specified')
        return {'Options':self.options, 'Dataset':self.dataset}    

class CheckDatasetExists(Task):
    """Use 'datasets.py' to check that the dataset exists in the production system.
    """
    def __init__(self, dataset, user, options):
        Task.__init__(self,'CheckDatasetExists', dataset, user, options)
    def run(self, input):
        pattern = fnmatch.translate(self.options.wildcard)
        run_range = (self.options.min_run, self.options.max_run)
        data = createDataset(self.user, self.dataset, pattern, run_range = run_range)
        if( len(data.listOfGoodFiles()) == 0 ):
            raise Exception('no good root file in dataset %s | %s | %s | %s' % (self.user,
                                                                            self.dataset,
                                                                            self.options.wildcard,
                                                                            run_range))        
        return {'Dataset':self.dataset}

class BaseDataset(Task):
    """Query DAS to find dataset name in DBS - see https://cmsweb.cern.ch/das/"""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'BaseDataset', dataset, user, options)
    def addOption(self, parser):
        parser.add_option("-n", "--name", dest="name", default=None,help='The name of the dataset in DAS. Will be guessed if not specified')
    def query(self, dataset):
        """Query DAS to find out how many events are in the dataset"""

        host    = self.options.host
        debug   = self.options.verbose
        idx     = self.options.idx
        limit   = self.options.limit
        
        def check(ds):
            query = 'dataset=%s' % ds
            result = Das.get_data(host, query, idx, limit, debug)
            result = result.replace('null','None')
            result = result.replace('true','True')
            result = result.replace('false','False')
            data = eval(result)
            if data['status'] != 'ok':
                raise Exception("Das query failed: Output is '%s'" % data)
            return (data['data'],data)

        data = None
        exists = False
        
        if self.options.name is None:
            #guess the dataset name in DBS
            tokens = [t for t in dataset.split(os.sep) if t]
            if len(tokens) >= 3:
                #DBS names always have three entries
                ds = os.sep + os.sep.join(tokens[0:3])
                if ds:
                    exists, data = check(ds)
                    self.options.name = ds
        else:
            exists, data = check(self.options.name)
            if not exists:
                raise Exception("Specified dataset '%s' not found in Das. Please check." % self.options.name)
        
        if data is None:
            raise Exception("Dataset '%s' not found in Das. Please check." % self.dataset)
        return data
    
    def run(self, input):
        output = {}
        if (hasattr(self.options,'check') and self.options.check) or not hasattr(self.options,'check'):
            output = self.query(self.dataset)
        return {'Name':self.options.name,'Das':output}

class GZipFiles(Task):
    """GZip a list of files"""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'GZipFiles', dataset, user, options)
    def gzip(self, fileName):
        output = '%s.gz' % fileName
        
        f_in = open(fileName, 'rb')
        f_out = gzip.open(output, 'wb')
        f_out.writelines(f_in)
        f_out.close()
        f_in.close()
        #remove the original file once we've gzipped it
        os.remove(fileName)
        return output
           
    def run(self, input):
        files = input['FilesToCompress']['Files']
        
        compressed = []
        for f in files:
            if f is None or not f: continue
            if os.path.exists(f):
                gz = self.gzip(f)
                compressed.append(gz)
        return {'CompressedFiles':compressed}
    
class CleanFiles(Task):
    """Remove a list of files"""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'CleanFiles', dataset, user, options)
    def run(self, input):
        files = input['FilesToClean']['Files']
        removed = []
        for f in files:
            if f is None or not f: continue
            if os.path.exists(f): os.remove(f)
            removed.append(f)
        return {'CleanedFiles':removed}

class FindOnCastor(Task):
    """Checks that the sample specified exists in the CASTOR area of the user specified. The directory must exist."""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'FindOnCastor', dataset, user, options)
    def run(self, input):
        if self.user == 'CMS':
            return {'Topdir':None,'Directory':None}
        topdir = castortools.lfnToCastor(castorBaseDir(user=self.user))
        directory = '%s/%s' % (topdir,self.dataset)
        # directory = directory.replace('//','/')
        if not castortools.fileExists(directory):
            if hasattr(self,'create') and self.create:
                castortools.createCastorDir(directory)
                #castortools.chmod(directory,'775')
        if not castortools.isDirectory(directory): 
            raise Exception("Dataset directory '%s' does not exist or could not be created" % directory)
        return {'Topdir':topdir,'Directory':directory}  

class CheckForMask(Task):
    """Tests if a file mask, created by edmIntegrityCheck.py, is present already and reads it if so."""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'CheckForMask', dataset, user, options)
    def addOption(self, parser):
        parser.add_option("-c", "--check", dest="check", default=False, action='store_true',help='Check filemask if available')        
    def run(self, input):
        #skip for DBS
        if self.user == 'CMS':
            return {'MaskPresent':True,'Report':'Files taken from DBS'}
        
        dir = input['FindOnCastor']['Directory']
        mask = "IntegrityCheck"
        file_mask = []  

        report = None
        if (hasattr(self.options,'check') and self.options.check) or not hasattr(self.options,'check'):
            file_mask = castortools.matchingFiles(dir, '^%s_.*\.txt$' % mask)

            if file_mask:
                p = PublishToFileSystem(mask)
                report = p.get(dir)
        return {'MaskPresent':report is not None,'Report':report}
    
class CheckForWrite(Task):
    """Checks whether you have write access to the CASTOR directory specified"""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'CheckForWrite', dataset, user, options)
    def run(self, input):
        """Check that the directory is writable"""
        if self.user == 'CMS':
            return {'Directory':None,'WriteAccess':True}
        dir = input['FindOnCastor']['Directory']
        if self.options.check:
            
            _, name = tempfile.mkstemp('.txt',text=True)
            testFile = file(name,'w')
            testFile.write('Test file')
            testFile.close()

            store = castortools.castorToLFN(dir) 
            #this is bad, but castortools is giving me problems
            if not os.system('cmsStage %s %s' % (name,store)):
                fname = '%s/%s' % (dir,os.path.basename(name))
                write = castortools.fileExists(fname)
                if write:
                    castortools.rm(fname)
                else:
                    raise Exception("Failed to write to directory '%s'" % dir)
            os.remove(name)
        return {'Directory':dir,'WriteAccess':True}
    
class GenerateMask(Task):
    """Uses edmIntegrityCheck.py to generate a file mask for the sample if one is not already present."""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'GenerateMask', dataset, user, options)
    def addOption(self, parser):
        parser.add_option("-r", "--recursive", dest="resursive", default=False, action='store_true',help='Walk the mass storage device recursively')
        parser.add_option("-p", "--printout", dest="printout", default=False, action='store_true',help='Print a report to stdout')        
    def run(self, input):
        
        report = None
        if self.options.check and not input['CheckForMask']['MaskPresent']:
            
            options = copy.deepcopy(self.options)
            options.user = self.user

            if input.has_key('BaseDataset'):
                options.name = input['BaseDataset']['Name']
            else:
                options.name = None
            
            check = IntegrityCheck(self.dataset,options)
            check.test()
            report = check.structured()
            pub = PublishToFileSystem(check)
            pub.publish(report)
        elif input['CheckForMask']['MaskPresent']:
            report = input['CheckForMask']['Report']
        
        return {'MaskPresent':report is not None,'Report':report}

class CreateJobDirectory(Task):
    """Generates a job directory on your local drive"""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'CreateJobDirectory', dataset, user, options)
    def addOption(self, parser):
        parser.add_option("-o","--output", dest="output", default=None,help='The directory to use locally for job files')           
    def run(self, input):
        if self.options.output is not None:
            output = self.options.output
        else:
            # output = '%s_%s' % (self.dataset.replace('/','.'),datetime.datetime.now().strftime("%s"))
            # if output.startswith('.'):
            output = '%s_%s' % (self.dataset,datetime.datetime.now().strftime("%s"))
            output = output.lstrip('/')
        if not os.path.exists(output):
            mkdir_p(output)
        return {'JobDir':output,'PWD':os.getcwd()}

class SourceCFG(Task):
    """Generate a source CFG using 'sourceFileList.py' by listing the CASTOR directory specified. Applies the file wildcard, '--wildcard'"""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'SourceCFG', dataset, user, options)    
    def addOption(self, parser):
        parser.add_option("--min-run", dest="min_run", default=-1, type=int, help='When querying DBS, require runs >= than this run')
        parser.add_option("--max-run", dest="max_run", default=-1, type=int, help='When querying DBS, require runs <= than this run')
        parser.add_option("--input-prescale", dest="prescale", default=1, type=int, help='Randomly prescale the number of good files by this factor.')
    def run(self, input):

        jobdir = input['CreateJobDirectory']['JobDir']
        pattern = fnmatch.translate(self.options.wildcard)
        
        run_range = (self.options.min_run, self.options.max_run)
        data = createDataset(self.user, self.dataset, pattern, run_range = run_range)
        good_files = data.listOfGoodFilesWithPrescale(self.options.prescale)
        #will mark prescale removed files as bad in comments
        bad_files = [fname for fname in data.listOfFiles() if not fname in good_files]
        
        source = os.path.join(jobdir,'source_cfg.py')
        output = file(source,'w')
        output.write('###SourceCFG:\t%d GoodFiles; %d BadFiles found in mask; Input prescale factor %d\n' % (len(good_files),len(bad_files),self.options.prescale) )
        output.write('files = ' + str(good_files) + '\n')
        for bad_file in bad_files:
            output.write("###SourceCFG:\tBadInMask '%s'\n" % bad_file)
        output.close()
        return {'SourceCFG':source}


def insertLines( insertedTo, toInsert ):
    '''insert a sequence in another sequence.

    the sequence is inserted either at the end, or at the position
    of the HOOK, if it is found.
    The HOOK is considered as being found if
      str(elem).find(###ProductionTaskHook$$$)
    is true for one of the elements in the insertedTo sequence. 
    '''
    HOOK = '###ProductionTaskHook$$$'
    hookIndex = None
    for index, line in enumerate(insertedTo):
        line = str(line)
        if line.find(HOOK)>-1:
            hookIndex = index
            break        
    if hookIndex is not None:
        before = insertedTo[:hookIndex]
        after = insertedTo[hookIndex:]
        result = before + toInsert + after
        return result
    else:
        insertedTo.extend( toInsert )
        return insertedTo

    
class FullCFG(Task):
    """Generate the full CFG needed to run the job and writes it to the job directory"""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'FullCFG', dataset, user, options)
    def addOption(self, parser):
        parser.add_option("--cfg", dest="cfg", default=None, help='The top level CFG to run')           
        parser.add_option("--nEventsPerJob", dest="nEventsPerJob", default=None, help='Number of events per job (for testing)')           
    def run(self, input):
        
        jobdir = input['CreateJobDirectory']['JobDir']

        if self.options.cfg is None or not os.path.exists(self.options.cfg):
            raise Exception("The file '%s' does not exist. Please check." % self.options.cfg)
        
        config = file(self.options.cfg).readlines()
        sourceFile = os.path.basename(input['SourceCFG']['SourceCFG'])
        if sourceFile.lower().endswith('.py'):
            sourceFile = sourceFile[:-3]
        
        source = os.path.join(jobdir,'full_cfg.py')
        output = file(source,'w')

        nEventsPerJob = -1
        if self.options.nEventsPerJob:
            nEventsPerJob = int(self.options.nEventsPerJob)
        
        toInsert = ['\nfrom %s import *\n' % sourceFile,
                    'process.source.fileNames = files\n',
                    'if hasattr(process,"maxEvents"): process.maxEvents.input = cms.untracked.int32({nEvents})\n'.format(nEvents=nEventsPerJob),
                    'if hasattr(process,"maxLuminosityBlocks"): process.maxLuminosityBlocks.input = cms.untracked.int32(-1)\n'
                    'datasetInfo = ("%s","%s","%s")\n' % (self.user, self.dataset, fnmatch.translate(self.options.wildcard) )
                    ]
        config = insertLines( config, toInsert )
        output.writelines(config)
        output.close()
        return {'FullCFG':source}

class CheckConfig(Task):
    """Check the basic syntax of a CFG file by running python on it."""    
    def __init__(self, dataset, user, options):
        Task.__init__(self,'CheckConfig', dataset, user, options)
    def run(self, input):
        
        full = input['FullCFG']['FullCFG']
        
        child = subprocess.Popen(['python',full], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        stdout, stderr = child.communicate()
        if child.returncode != 0:
            raise Exception("Syntax check of cfg failed. Error was '%s'. (%i)" % (stderr,child.returncode))
        return {'Status':'VALID'}   

class RunTestEvents(Task):
    """Run cmsRun but with a small number of events on the job CFG."""    

    def __init__(self, dataset, user, options):
        Task.__init__(self,'RunTestEvents', dataset, user, options)
    def run(self, input):
        
        full = input['FullCFG']['FullCFG']
        jobdir = input['CreateJobDirectory']['JobDir']
        
        config = file(full).readlines()
        source = os.path.join(jobdir,'test_cfg.py')
        output = file(source,'w')
        toInsert = ['\n',
                    'process.maxEvents.input = cms.untracked.int32(5)\n',
                    'if hasattr(process,"source"): process.source.fileNames = process.source.fileNames[:10]\n'
                    ]
        config = insertLines( config, toInsert )
        output.writelines(config)
        output.close()
                
        pwd = os.getcwd()
        
        error = None
        try:
            os.chdir(jobdir)
            
            child = subprocess.Popen(['cmsRun',os.path.basename(source)], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            stdout, stderr = child.communicate()
            
            if child.returncode != 0:
                error = "Failed to cmsRun with a few events. Error was '%s' (%i)." % (stderr,child.returncode)
        finally:
            os.chdir(pwd)
            
        if error is not None:
            raise Exception(error)

        return {'Status':'VALID','TestCFG':source}
    
class ExpandConfig(Task):
    """Runs edmConfigDump to produce an expanded cfg file"""    

    def __init__(self, dataset, user, options):
        Task.__init__(self,'ExpandConfig', dataset, user, options)
    def run(self, input):
        
        full = input['FullCFG']['FullCFG']
        jobdir = input['CreateJobDirectory']['JobDir']

        config = file(full).read()
        source = os.path.join(jobdir,'test_cfg.py')
        expanded = 'Expanded%s' % os.path.basename(full)
        output = file(source,'w')
        output.write(config)
        output.write("file('%s','w').write(process.dumpPython())\n" % expanded)
        output.close()

        pwd = os.getcwd()
        
        result = {}
        error = None
        try:
            os.chdir(jobdir)
            
            child = subprocess.Popen(['python',os.path.basename(source)], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            stdout, stderr = child.communicate()
            
            if child.returncode != 0:
                error = "Failed to edmConfigDump. Error was '%s' (%i)." % (stderr,child.returncode)
            result['ExpandedFullCFG'] = os.path.join(jobdir,expanded)
            
        finally:
            os.chdir(pwd)
            
        if error is not None:
            raise Exception(error)

        return result
    
class WriteToDatasets(Task):
    """Publish the sample to 'Datasets.txt' if required"""    
    def __init__(self, dataset, user, options):
        Task.__init__(self,'WriteToDatasets', dataset, user, options)
    def run(self, input):
        name = "%s/%s" % (self.dataset,self.options.tier)
        name = name.replace('//','/')
        user = self.options.batch_user
        added = addToDatasets(name, user = user)
        return {'Added':added, 'Name':name, 'User':user}       
    
class RunCMSBatch(Task):
    """Run the 'cmsBatch.py' command on your CFG, submitting to the CERN batch system"""    

    def __init__(self, dataset, user, options):
        Task.__init__(self,'RunCMSBatch', dataset, user, options)
    def addOption(self, parser):
        parser.add_option("--batch_user", dest="batch_user", help="The user for LSF", default=os.getlogin())
        parser.add_option("--run_batch", dest="run_batch", default=True, action='store_true',help='Run on the batch system')
        parser.add_option("-N", "--numberOfInputFiles", dest="nInput",help="Number of input files per job",default=5,type=int)
        parser.add_option("-q", "--queue", dest="queue", help="The LSF queue to use", default="1nh")        
        parser.add_option("-t", "--tier", dest="tier",
                          help="Tier: extension you can give to specify you are doing a new production. If you give a Tier, your new files will appear in sampleName/tierName, which will constitute a new dataset.",
                          default="")
        parser.add_option("-G", "--group", dest="group", help="The LSF user group to use, e.g. 'u_zh'", default=None)        
        
    def run(self, input):
        find = FindOnCastor(self.dataset,self.options.batch_user,self.options)
        find.create = True
        out = find.run({})
        
        full = input['ExpandConfig']['ExpandedFullCFG']
        jobdir = input['CreateJobDirectory']['JobDir']
        
        sampleDir = os.path.join(out['Directory'],self.options.tier)
        sampleDir = castortools.castorToLFN(sampleDir)
        
        cmd = ['cmsBatch.py',str(self.options.nInput),os.path.basename(full),'-o','%s_Jobs' % self.options.tier,'--force']
        cmd.extend(['-r',sampleDir])
        if self.options.run_batch:
            jname = "%s/%s" % (self.dataset,self.options.tier)
            jname = jname.replace("//","/")
            user_group = ''
            if self.options.group is not None:
                user_group = '-G %s' % self.options.group
            cmd.extend(['-b',"'bsub -q %s -J %s -u cmgtoolslsf@gmail.com %s < ./batchScript.sh | tee job_id.txt'" % (self.options.queue,jname,user_group)])
        print " ".join(cmd)
        
        pwd = os.getcwd()
        
        error = None
        try:
            os.chdir(jobdir)
            returncode = os.system(" ".join(cmd))

            if returncode != 0:
                error = "Running cmsBatch failed. Return code was %i." % returncode
        finally:
            os.chdir(pwd)
            
        if error is not None:
            raise Exception(error)

        return {'SampleDataset':"%s/%s" % (self.dataset,self.options.tier),'BatchUser':self.options.batch_user,
                'SampleOutputDir':sampleDir,'LSFJobsTopDir':os.path.join(jobdir,'%s_Jobs' % self.options.tier)}

class MonitorJobs(Task):
    """Monitor LSF jobs created with cmsBatch.py. Blocks until all jobs are finished."""    
    def __init__(self, dataset, user, options):
        Task.__init__(self,'MonitorJobs', dataset, user, options)
    
    def getjobid(self, job_dir):
        """Parse the LSF output to find the job id"""
        input = os.path.join(job_dir,'job_id.txt')
        result = None
        if os.path.exists(input):
            contents = file(input).read()
            for c in contents.split('\n'):
                if c and re.match('^Job <\\d*> is submitted to queue <.*>',c) is not None:
                    try:
                        result = c.split('<')[1].split('>')[0]
                    except Exception, e:
                        print >> sys.stderr, 'Job ID parsing error',str(e),c
        return result
    
    def monitor(self, jobs, previous):

        #executes bjobs with a list of job IDs
        cmd = ['bjobs','-u',self.options.batch_user]
        cmd.extend([v for v in jobs.values() if v is not None])#filter out unknown IDs
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        stdout, stderr = child.communicate()

        def parseHeader(header):
            """Parse the header from bjobs"""
            tokens = [t for t in header.split(' ') if t]
            result = {}
            for i in xrange(len(tokens)):
                result[tokens[i]] = i          

            return result

        result = {}
        if stdout:
            lines = stdout.split('\n')
            if lines:
                header = parseHeader(lines[0])                                 
                if not 'STAT' in header or not 'JOBID' in header:
                    print >> sys.stderr, 'Problem parsing bjobs header\n',lines
                    return result
                for line in lines[1:]:
                    #TODO: Unreliable for some fields, e.g. dates
                    tokens = [t for t in line.split(' ') if t]
                    if len(tokens) < len(header): continue
                    id = tokens[header['JOBID']]
                    user = tokens[header['USER']]
                    status = tokens[header['STAT']]

                    result[id] = status               
                    
        if stderr:
            lines = stderr.split('\n')
            if lines:
                for line in lines:
                    if line and re.match('^Job <\\d*> is not found',line) is not None:
                        try:
                            id = line.split('<')[1].split('>')[0]
                            if not result.has_key(id) and not previous.has_key(id):
                                result[id] = 'FORGOTTEN'
                        except Exception, e:
                            print >> sys.stderr, 'Job ID parsing error in STDERR',str(e),line
        
        #after one hour the status is no longer available     
        if result:
            for id in jobs.values():
                if not result.has_key(id) and previous.has_key(id):
                    result[id] = previous[id]
        return result
    
    def run(self, input):

        # return #COLIN
        jobsdir = input['RunCMSBatch']['LSFJobsTopDir']
        if not os.path.exists(jobsdir):
            raise Exception("LSF jobs dir does not exist: '%s'" % jobsdir)
        
        subjobs = [s for s in glob.glob("%s/Job_[0-9]*" % jobsdir) if os.path.isdir(s)]
        jobs = {}
        for s in subjobs:
            jobs[s] = self.getjobid(s)

        def checkStatus(stat):
            
            #gzip files on the fly
            actions = {'FilesToCompress':{'Files':[]}}
            
            result = {}
            for j, id in jobs.iteritems():
                if id is None:
                    result[j] = 'UNKNOWN'
                else:
                    if stat.has_key(id):
                        result[j] = stat[id]
                        if result[j] in ['DONE','EXIT','FORGOTTEN']:
                            stdout = os.path.join(j,'LSFJOB_%s' % id,'STDOUT')
                            if os.path.exists(stdout):
                                #compress this file
                                actions['FilesToCompress']['Files'].append(stdout)
                                result[j] = '%s.gz' % stdout
                            elif os.path.exists('%s.gz' % stdout):
                                result[j] = '%s.gz' % stdout
                            else:
                                result[j] = 'NOSTDOUT'
                        
                            #also compress the stderr, although this is mostly empty
                            stderr = os.path.join(j,'LSFJOB_%s' % id,'STDERR')
                            if os.path.exists(stderr):
                                #compress this file
                                actions['FilesToCompress']['Files'].append(stderr)
                                
            compress = GZipFiles(self.dataset,self.user,self.options)
            compress.run(actions)
            return result
        
        def countJobs(stat):
            """Count jobs that are monitorable - i.e. not in a final state"""
            result = []
            for j, id in jobs.iteritems():
                if id is not None and stat.has_key(id):
                    st = stat[id]
                    if st in ['PEND','PSUSP','RUN','USUSP','SSUSP','WAIT']:
                        result.append(id)
            return result
        
        def writeKillScript(mon):
            """Write a shell script to kill the jobs we know about"""
            kill = os.path.join(jobsdir,'kill_jobs.sh')
            output = file(kill,'w')
            script = """
#!/usr/bin/env bash
echo "Killing jobs"
bkill -u %s %s
            """ % (self.options.batch_user," ".join(mon))
            output.write(script)
            output.close()
            return mon
                    
        #continue monitoring while there are jobs to monitor
        status = self.monitor(jobs,{})
        monitorable = writeKillScript(countJobs(status))
        count = 0
        
        while monitorable:
            job_status = checkStatus(status)
            time.sleep(60)
            status = self.monitor(jobs,status)
            monitorable = writeKillScript(countJobs(status))
            if not (count % 3):
                print '%s: Monitoring %i jobs (%s)' % (self.name,len(monitorable),self.dataset)
            count += 1
            
        return {'LSFJobStatus':checkStatus(status),'LSFJobIDs':jobs}  
    
class CheckJobStatus(Task):
    """Checks the job STDOUT to catch common problems like exceptions, CPU time exceeded. Sets the job status in the report accordingly."""    
    def __init__(self, dataset, user, options):
        Task.__init__(self,'CheckJobStatus', dataset, user, options)
    def addOption(self, parser):
        parser.add_option("--output_wildcard", dest="output_wildcard", help="The wildcard to use when testing the output of this production (defaults to same as -w)", default=None)           
    def run(self, input):
        
        job_status = input['MonitorJobs']['LSFJobStatus']

        result = {}
        for j, status in job_status.iteritems():
            valid = True
            if os.path.exists(status):

                fileHandle = None
                if status.endswith('.gz') or status.endswith('.GZ'):
                    fileHandle = gzip.GzipFile(status)
                else:
                    fileHandle = file(status)
                
                open_count = 0
                close_count = 0
                for line in fileHandle:
                    #start by counting files opened and closed
                    #suggestion from Enrique
                    if 'pened file' in line:
                        open_count += 1
                    if 'losed file' in line:
                        close_count += 1
                    
                    if 'Exception' in line:
                        result[j] = 'Exception'
                        valid = False
                        break
                    elif 'CPU time limit exceeded' in line:
                        result[j] = 'CPUTimeExceeded'
                        valid = False
                        break
                    elif 'Killed' in line:
                        result[j] = 'JobKilled'
                        valid = False
                        break
                    elif 'A fatal system signal has occurred' in line:
                        result[j] = 'SegFault'
                        valid = False
                        break
                    
                if valid and open_count != close_count:
                    result[j] = 'FileOpenCloseMismatch'
                    valid = False
                if valid:
                    result[j] = 'VALID'
            else:
                result[j] = status

        #allows a different wildcard in the final check.
        options = copy.deepcopy(self.options)
        if self.options.output_wildcard is not None:
            options.wildcard = self.options.output_wildcard

        mask = GenerateMask(input['RunCMSBatch']['SampleDataset'],self.options.batch_user,options)
        report = mask.run({'CheckForMask':{'MaskPresent':False}})
        report['LSFJobStatusCheck'] = result
        return report
    
class WriteJobReport(Task):
    """Write a summary report on each job"""    
    def __init__(self, dataset, user, options):
        Task.__init__(self,'WriteJobReport', dataset, user, options)
    def run(self, input):
        
        report = input['CheckJobStatus']
        
        #collect a list of jobs by status
        states = {}
        for j, status in report['LSFJobStatusCheck'].iteritems():
            if not states.has_key(status):
                states[status] = []
            states[status].append(j)
        jobdir = input['CreateJobDirectory']['PWD']
        if not os.path.exists(jobdir):
            raise Exception("Top level job directory not found: '%s'" % jobdir)
        report_file = os.path.join(input['CreateJobDirectory']['JobDir'],'resubmit.sh')

        output = file(report_file,'w')
        output.write('#!/usr/bin/env bash\n')
        
        if report['MaskPresent']:
            mask = report['Report']
            output.write('#PrimaryDatasetFraction: %f\n' % mask['PrimaryDatasetFraction'])
            output.write('#FilesGood: %i\n' % mask['FilesGood'])
            output.write('#FilesBad: %i\n' % mask['FilesBad'])        
        
        user_group = ''
        if self.options.group is not None:
            user_group = '-G %s' % self.options.group

        for status, jobs in states.iteritems():
            output.write('# %d jobs found in state %s\n' % (len(jobs),status) )
            if status == 'VALID':
                continue
            for j in jobs:
                jdir = os.path.join(jobdir,j)
                output.write('pushd %s; bsub -q %s -J RESUB -u cmgtoolslsf@gmail.com %s < ./batchScript.sh | tee job_id_resub.txt; popd\n' % (jdir,self.options.queue,user_group))
        output.close()
            
        return {'SummaryFile':report_file}
    
class CleanJobFiles(Task):
    """Removes and compresses auto-generated files from the job directory to save space."""    
    def __init__(self, dataset, user, options):
        Task.__init__(self,'CleanJobFiles', dataset, user, options)
    def run(self, input):
        
        jobdir = input['CreateJobDirectory']['JobDir']
        jobs = input['MonitorJobs']['LSFJobIDs']
        job_status = input['MonitorJobs']['LSFJobStatus']
        
        actions = {'FilesToCompress':{'Files':[]},'FilesToClean':{'Files':[]}}
        
        actions['FilesToClean']['Files'].append(input['ExpandConfig']['ExpandedFullCFG'])
        if input.has_key('RunTestEvents'):
            actions['FilesToClean']['Files'].append(input['RunTestEvents']['TestCFG'])

        for rt in glob.iglob('%s/*.root' % jobdir):
            actions['FilesToClean']['Files'].append(rt)
        for pyc in glob.iglob('%s/*.pyc' % jobdir):
            actions['FilesToClean']['Files'].append(pyc)

        for j in jobs:
            status = job_status[j]
            if os.path.exists(status) and not status.endswith('.gz'):
                actions['FilesToCompress']['Files'].append(status)
                
        compress = GZipFiles(self.dataset,self.user,self.options)
        compressed = compress.run(actions)

        clean = CleanFiles(self.dataset,self.user,self.options)
        removed = clean.run(actions)
        return {'Cleaned':removed,'Compressed':compressed}
    

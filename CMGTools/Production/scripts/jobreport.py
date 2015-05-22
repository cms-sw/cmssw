#!/bin/env python

import sys, os, re, gzip, fnmatch, tarfile, tempfile
import CMGTools.Production.eostools as eostools
from CMGTools.Production.dataset import createDataset

def checkForLogger( dataset_lfn_dir ):
	"""Checks the EOS directory for a Logger.tgz file, if not found, escapes
	'sampleName' takes the name of the sample as a string
	'fileOwner' takes the file owner on EOS as a string
	"""
	if len( eostools.matchingFiles( dataset_lfn_dir, "Logger.tgz" ) )  == 1:
		return createLoggerTemporaryFile( dataset_lfn_dir )
	else: 
		raise NameError("ERROR: No Logger.tgz file found for this sample. If you would like to preceed anyway, please copy Logger.tgz from your local production directory to your production directory on eos.\n")
	
def createLoggerTemporaryFile( dataset_lfn_dir ):
	"""Build a temporary logger file object and tarfile object to be used 
	when retrieving tags and jobs"""
	try:
		logger_file = tempfile.NamedTemporaryFile()
		os.system("cmsStage -f "+ os.path.join( eostools.eosToLFN(dataset_lfn_dir), "Logger.tgz") + " " + logger_file.name)
		logger_tar_object = tarfile.open(fileobj = logger_file)
		if len( logger_tar_object.getmembers() )==0: 
			print "\nERROR: Failed to stage logger file"
			exit(-1)
		return logger_tar_object
	except:
		print "\nERROR: Failed to stage logger file"
		exit(-1)

def buildBadJobsList( dataset ):
	badJobs = []
	try:
		# Open the file in the logger and get the value
		print dataset.lfnDir
		logger_tar_object = checkForLogger( dataset.lfnDir )
		nJobsFile = logger_tar_object.extractfile("Logger/logger_jobs.txt") #extract Logger/logger_jobs.txt if it exists    
		nJobs = int(nJobsFile.read().split(": ")[1].split("\n")[0])         #read job number from file
	except:
		print "ERROR: No jobs file found in logger" 
		exit( -1 )

	if nJobs == None:
		print "ERROR:Invalid job number - Corrupt jobs report from Logger/logger_jobs.txt"
		exit( -1 )
	else:
		goodFiles = data.listOfGoodFiles()
		goodJobNumbers =  sorted( map( jobNumber, goodFiles ) )
		totalJobNumbers = range( 1, nJobs )
	
		badJobs = list( set(totalJobNumbers) - set(goodJobNumbers) ) 
	return badJobs
		
def jobNumber( fileName ):
    pattern = re.compile('.*_(\d+)\.root$')
    m = pattern.match(fileName)
    return int(m.group(1))

def jobDir( allJobsDir, job ):
    return '{all}/Job_{job}'.format(all=allJobsDir, job=job)

def lsfReport( stdoutgz, unzip=False, nLines=100):
    sep_line = '-'*70
    print
    print sep_line
    print stdoutgz
    print
    stdout = None
    if unzip:
        stdout = gzip.open(stdoutgz)
    else:
        stdout = open(stdoutgz)
    lines = stdout.readlines()
    nLines = min(nLines, len(lines))
    for line in lines[-nLines:]:
        line = line.rstrip('\n')
        print line

def jobReport( allJobsDir, job, nLines=100):
    jdir = jobDir( allJobsDir, job )
    for root, dirs, files in os.walk(jdir):
        stdout = 'STDOUT.gz'
        if stdout in files:
            lsfReport('/'.join( [root, stdout] ), True, nLines)
        stdout = 'STDOUT'       
        if stdout in files:
            lsfReport('/'.join( [root, stdout] ), False, nLines)

def jobSubmit( allJobsDir, job, cmd):    
    jdir = jobDir( allJobsDir, job )
    oldPwd = os.getcwd()
    os.chdir( jdir )
    print cmd
    os.system( cmd )
    os.chdir( oldPwd )


    
if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = """
    %prog [options] <dataset> <jobs_dir>
    
    Prints the list of bad jobs.
    Using the options, you can get a log of what happened during each bad job,
    and you can resubmit these jobs.
    """
    parser.add_option("-r", "--report", dest="report",
                      action = 'store_true',
                      default=False,
                      help='Print report for bad jobs.')
    parser.add_option("-n", "--nlines", dest="nlines",
                      default=100,
                      help='Number of lines in the report for each job.')
    parser.add_option("-s", "--submit", dest="submit",
                      action = 'store_true',
                      default=False,
                      help='Print resubmission command')
    parser.add_option("-u", "--user", dest="user", default=os.environ.get('USER', None),help='user owning the dataset.\nInstead of the username, give "LOCAL" to read datasets in a standard unix filesystem, and "CMS" to read official CMS datasets present at CERN.')
    parser.add_option("-w", "--wildcard", dest="wildcard", default='*root',help='A UNIX style wildcard for root file printout')
    parser.add_option("-b", "--batch", dest="batch",
                      help="batch command. default is: 'bsub -q 8nh < batchScript.sh'. You can also use 'nohup < ./batchScript.sh &' to run locally.",
                      default="bsub -q 8nh < ./batchScript.sh")
    parser.add_option("-c", "--readcache", dest="readcache",
                      action = 'store_true',
                      default=False,
                      help='Read from the cache.')
    parser.add_option("-j", "--badjobs", dest="badjoblists",
                      default=None,
                      help='Lists of bad jobs, as [1,5];[2,5,7]')
    
    (options,args) = parser.parse_args()

    if len(args)!=2:
        print 'please provide the dataset name and the job directory in argument'
        sys.exit(1)
    
    dataset = args[0]
    allJobsDir = args[1]

    user = options.user
    pattern = fnmatch.translate( options.wildcard )

    data = createDataset(user, dataset, pattern, options.readcache)
    

    badJobs = []
    if options.badjoblists is None:
        badJobs = buildBadJobsList( data )

    else:
        # import pdb; pdb.set_trace()
        bjlsstr = options.badjoblists.split(';')
        bjlsstr = filter(lambda x: len(x)>0, bjlsstr)
        bjls = map(eval, bjlsstr)
        setOfBadJobs = set()
        for bjl in bjls:
            setOfBadJobs.update( set(bjl) )
        # print setOfBadJobs
        # sys.exit(1)
        # print len(badJobs), 'bad jobs'
        # print badJobs
        badJobs = sorted( setOfBadJobs )

    if options.report:
        for job in badJobs:
            jobReport(allJobsDir, job, int(options.nlines) )
    elif options.submit:
        for job in badJobs:
            jobSubmit(allJobsDir, job, options.batch)
    else:
        print badJobs

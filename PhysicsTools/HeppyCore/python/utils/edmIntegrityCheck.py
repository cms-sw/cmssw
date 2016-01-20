#!/usr/bin/env python
"""
Classes to check that a set of ROOT files are OK and publish a report
"""

import datetime, fnmatch, json, os, shutil, sys, tempfile, time
import subprocess

import eostools as castortools
from timeout import timed_out, TimedOutExc
from castorBaseDir import castorBaseDir
from dataset import CMSDataset

class PublishToFileSystem(object):
    """Write a report to storage"""
    
    def __init__(self, parent):
        if type(parent) == type(""):
            self.parent = parent
        else:
            self.parent = parent.__class__.__name__
    
    def publish(self, report):
        """Publish a file"""
        for path in report['PathList']:
            _, name = tempfile.mkstemp('.txt', text=True)
            json.dump(report, file(name,'w'), sort_keys=True, indent=4)
            
            fname = '%s_%s.txt' % (self.parent, report['DateCreated'])
            #rename the file locally - TODO: This is a potential problem
            nname = os.path.join(os.path.dirname(name),fname)
            os.rename(name, nname)
            
            castor_path = castortools.lfnToCastor(path)
            new_name = '%s/%s' % (castor_path, fname)
            castortools.xrdcp(nname,path)
            time.sleep(1)
            
            if castortools.fileExists(new_name):
                
                #castortools.move(old_name, new_name)
                #castortools.chmod(new_name, '644')

                print "File published: '%s'" % castortools.castorToLFN(new_name)
                os.remove(nname)
            else:
                pathhash = path.replace('/','.')
                hashed_name = 'PublishToFileSystem-%s-%s' % (pathhash, fname)
                shutil.move(nname, hashed_name)
                print >> sys.stderr, "Cannot write to directory '%s' - written to local file '%s' instead." % (castor_path, hashed_name)
                
    def read(self, lfn, local = False):
        """Reads a report from storage"""
        if local:
            cat = file(lfn).read()
        else:
            cat = castortools.cat(castortools.lfnToCastor(lfn))
        #print "the cat is: ", cat
        return json.loads(cat)
    
    def get(self, dir):
        """Finds the lastest file and reads it"""
        reg = '^%s_.*\.txt$' % self.parent
        files = castortools.matchingFiles(dir, reg)
        files = sorted([ (os.path.basename(f), f) for f in files])
        if not files:
            return None
        return self.read(files[-1][1])
                

class IntegrityCheck(object):
    
    def __init__(self, dataset, options):
        if not dataset.startswith(os.sep):
            dataset = os.sep + dataset

        self.dataset = dataset
        self.options = options
        self.topdir = castortools.lfnToCastor( castorBaseDir(user=options.user) )
        self.directory = os.path.join(self.topdir, *self.dataset.split(os.sep))
        
        #event counters
        self.eventsTotal = -1
        self.eventsSeen = 0
        
        self.test_result = None
    
    def query(self):
        """Query DAS to find out how many events are in the dataset"""
        from production_tasks import BaseDataset
        base = BaseDataset(self.dataset, self.options.user, self.options)

        data = None
        output = base.run({})
        if output.has_key('Das'):
            self.options.name = output['Name']
            data = output['Das']
            
        if data is None:
            raise Exception("Dataset '%s' not found in Das. Please check." % self.dataset)
        #get the number of events in the dataset
        self.eventsTotal = CMSDataset.findPrimaryDatasetEntries(self.options.name, self.options.min_run, self.options.max_run)
    
    def stripDuplicates(self):
        
        import re
        
        filemask = {}
        for dirname, files in self.test_result.iteritems():
            for name, status in files.iteritems():
                fname = os.path.join(dirname, name)
                filemask[fname] = status
        
        def isCrabFile(name):
            _, fname = os.path.split(name)
            base, _ = os.path.splitext(fname)
            return re.match(".*_\d+_\d+_\w+$", base) is not None, base
        def getCrabIndex(base):
            tokens = base.split('_')
            if len(tokens) > 2:
                return (int(tokens[-3]), int(tokens[-2]))
            return None
            
        files = {}
        
        mmin = 1000000000
        mmax = -100000000
        for f in filemask:
            isCrab, base = isCrabFile(f)
            if isCrab:
                index = getCrabIndex(base)
                if index is not None:
                    jobid, retry = index
                    
                    mmin = min(mmin, jobid)
                    mmax = max(mmax, jobid)
                    if files.has_key(jobid) and filemask[f][0]:
                        files[jobid].append((retry, f))
                    elif filemask[f][0]:
                        files[jobid] = [(retry, f)]

        good_duplicates = {}
        bad_jobs = set()
        sum_dup = 0
        for i in xrange(mmin, mmax+1):
            if files.has_key(i):
                duplicates = files[i]
                duplicates.sort()

                fname = duplicates[-1][1]
                if len(duplicates) > 1:
                    for d in duplicates[:-1]:
                        good_duplicates[d[1]] = filemask[d[1]][1]
                        sum_dup += good_duplicates[d[1]]
            else:
                bad_jobs.add(i)
        return good_duplicates, sorted(list(bad_jobs)), sum_dup
    
    def test(self, previous = None, timeout = -1):
        if not castortools.fileExists(self.directory):
            raise Exception("The top level directory '%s' for this dataset does not exist" % self.directory)

        self.query()

        test_results = {}

        #support updating to speed things up
        prev_results = {}
        if previous is not None:
            for name, status in previous['Files'].iteritems():
                prev_results[name] = status
        
        filesToTest = self.sortByBaseDir(self.listRootFiles(self.directory))
        for dir, filelist in filesToTest.iteritems():
            filemask = {}
            #apply a UNIX wildcard if specified
            filtered = filelist
            if self.options.wildcard is not None:
                filtered = fnmatch.filter(filelist, self.options.wildcard)
                if not filtered:
                    print >> sys.stderr, "Warning: The wildcard '%s' does not match any files in '%s'. Please check you are using quotes." % (self.options.wildcard,self.directory)

            count = 0
            for ff in filtered:
                fname = os.path.join(dir, ff)
                lfn = castortools.castorToLFN(fname)
                
                #try to update from the previous result if available 
                if lfn in prev_results and prev_results[lfn][0]:
                    if self.options.printout:
                        print '[%i/%i]\t Skipping %s...' % (count, len(filtered),fname),
                    OK, num = prev_results[lfn]
                else:
                    if self.options.printout:
                        print '[%i/%i]\t Checking %s...' % (count, len(filtered),fname),
                    OK, num = self.testFileTimeOut(lfn, timeout)

                filemask[ff] = (OK,num)
                if self.options.printout:
                    print (OK, num)
                if OK:
                    self.eventsSeen += num
                count += 1
            test_results[castortools.castorToLFN(dir)] = filemask
        self.test_result = test_results

        self.duplicates, self.bad_jobs, sum_dup = self.stripDuplicates()
        #remove duplicate entries from the event count
        self.eventsSeen -= sum_dup
    
    def report(self):
        
        if self.test_result is None:
            self.test()
            
        print 'DBS Dataset name: %s' % self.options.name
        print 'Storage path: %s' % self.topdir
        
        for dirname, files in self.test_result.iteritems():
            print 'Directory: %s' % dirname
            for name, status in files.iteritems():
                fname = os.path.join(dirname, name)
                if not fname in self.duplicates:
                    print '\t\t %s: %s' % (name, str(status))
                else:
                    print '\t\t %s: %s (Valid duplicate)' % (name, str(status))
        print 'Total entries in DBS: %i' % self.eventsTotal
        print 'Total entries in processed files: %i' % self.eventsSeen
        if self.eventsTotal>0:
            print 'Fraction of dataset processed: %f' % (self.eventsSeen/(1.*self.eventsTotal))
        else:
            print 'Total entries in DBS not determined' 
        if self.bad_jobs:
            print "Bad Crab Jobs: '%s'" % ','.join([str(j) for j in self.bad_jobs])
        
    def structured(self):
        
        if self.test_result is None:
            self.test()
        
        totalGood = 0
        totalBad = 0

        report = {'data':{},
                  'ReportVersion':3,
                  'PrimaryDataset':self.options.name,
                  'Name':self.dataset,
                  'PhysicsGroup':'CMG',
                  'Status':'VALID',
                  'TierList':[],
                  'AlgoList':[],
                  'RunList':[],
                  'PathList':[],
                  'Topdir':self.topdir,
                  'StageHost':self.stageHost(),
                  'CreatedBy':self.options.user,
                  'DateCreated':datetime.datetime.now().strftime("%s"),
                  'Files':{}}
        
        for dirname, files in self.test_result.iteritems():
            report['PathList'].append(dirname)
            for name, status in files.iteritems():
                fname = os.path.join(dirname, name)
                report['Files'][fname] = status
                if status[0]:
                    totalGood += 1
                else:
                    totalBad += 1
                
        report['PrimaryDatasetEntries'] = self.eventsTotal
        if self.eventsTotal>0:
            report['PrimaryDatasetFraction'] = (self.eventsSeen/(1.*self.eventsTotal))
        else:
            report['PrimaryDatasetFraction'] = -1.
        report['FilesEntries'] = self.eventsSeen

        report['FilesGood'] = totalGood
        report['FilesBad'] = totalBad
        report['FilesCount'] = totalGood + totalBad
        
        report['BadJobs'] = self.bad_jobs
        report['ValidDuplicates'] = self.duplicates
        
        report['MinRun'] = self.options.min_run
        report['MaxRun'] = self.options.max_run

        return report
    
    def stageHost(self):
        """Returns the CASTOR instance to use"""
        return os.environ.get('STAGE_HOST','castorcms')

    def listFiles(self,dir):
        """Recursively list a file or directory on castor"""
        return castortools.listFiles(dir,self.options.resursive)

    def listRootFiles(self,dir):
        """filter out filenames so that they only contain root files"""
        return [f for f in self.listFiles(dir) if f.lower().endswith('.root')]

    def sortByBaseDir(self,files):
        """Sort files into directories"""
        result = {}
        for f in files:
            dirname = os.path.dirname(f)
            filename = os.path.basename(f)
            if not result.has_key(dirname): result[dirname] = []
            result[dirname].append(filename)
        return result


    def getParseNumberOfEvents(self,output):
        """Parse the output of edmFileUtil to get the number of events found"""
        tokens = output.split(' ')
        result = -2
        try:
            result = int(tokens[-4])
        except ValueError:
            pass
        return result

    def testFile(self,lfn):
        stdout = subprocess.Popen(['edmFileUtil',lfn], stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
        for error in ["Fatal Root Error","Could not open file","Not a valid collection"]:
            if error in stdout: return (False,-1)
        return (True,self.getParseNumberOfEvents(stdout))
    
    def testFileTimeOut(self,lfn, timeout):
        @timed_out(timeout)
        def tf(lfn):
            try:
                return self.testFile(lfn)
            except TimedOutExc, e:
                print >> sys.stderr, "ERROR:\tedmFileUtil timed out for lfn '%s' (%d)" % (lfn,timeout)
                return (False,-1)
        if timeout > 0:
            return tf(lfn)
        else:
            return self.testFile(lfn)



if __name__ == '__main__':
    
    pub = PublishToFileSystem('Test')
    report = {'DateCreated':'123456','PathList':['/store/cmst3/user/wreece']}
    pub.publish(report)
    print pub.get('/store/cmst3/user/wreece')

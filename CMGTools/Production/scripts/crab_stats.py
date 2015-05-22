#!/usr/bin/env python

import os
import random
import re
import subprocess
import sys

from xml.etree.ElementTree import ElementTree

class CrabStatus(object):

    def _getXMLReport(self, report_name):

        cmd = ['crab','-status','-USER.xml_report',report_name]
        stdout, stderr = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

        if stderr:
            print >>sys.stderr, stderr
            print >>sys.stderr, stdout

        for line in stdout.split('\n'):
            if line.startswith('Log file is'):
                tokens = line.split(' ')
                if tokens and os.path.exists(tokens[-1]):
                    log = tokens[-1]
                    tokens = [t for t in log.split(os.sep) if t]
                    base_dir = os.path.join(*tokens[:-2])
                    if log.startswith(os.sep) and not base_dir.startswith(os.sep):
                        base_dir = os.sep + base_dir
                    xml_file = os.path.join(base_dir,'share',report_name)
                    if os.path.exists(xml_file):
                        return xml_file
        return None

            

    def __init__(self):

        self.unique_name = 'Report_%s.xml' % random.randint(0,1000000)
        self.xml_file = self._getXMLReport(self.unique_name)
        if self.xml_file is None:
            raise Exception('Error: No XML report file found')

        self.tree = ElementTree()
        self.tree.parse(self.xml_file)

    def parseJobs(self):

        self.job_info = {}
        self.job_count = 0

        self.sub_count = {}
        
        jobs = self.tree.find('TaskJobs')
        for job in jobs.getiterator('RunningJob'):
            closed = job.get('closed')
            id = job.get('jobId')
            ret = job.get('applicationReturnCode')
            aret = ret = job.get('wrapperReturnCode')
            processStatus = job.get('processStatus')
            scheduleStatus = job.get('statusScheduler')
            sub = int(job.get('submission',0))

            status = '%s_%s_%s_%s' % (processStatus,scheduleStatus,ret,aret)
            if self.job_info.has_key(status):
                self.job_info[status].append(id)
            else:
                self.job_info[status] = [id]

            if self.sub_count.has_key(sub):
                self.sub_count[sub] += 1
            else:
                self.sub_count[sub] = 1

            self.job_count += 1


        self.job_info

    def printJobInfo(self):
    
        for key, job_list in self.job_info.iteritems():
            print '#',key,len(job_list),len(job_list)/(1.*self.job_count)
        print '# Submission counts: %s' % str(self.sub_count)


    def resubmit(self, status = 'created_Cleared_[0-9]+_[0-9]+'):

        resub = []
        sub = []
        match = []

        for key, job_list in self.job_info.iteritems():
            tokens = key.split('_')
            if re.match(status,key) is not None:
                if not tokens[-1] == '0' or not tokens[-2] == '0':
                    resub.extend(job_list)
            elif tokens and tokens[1] == 'Aborted':
                resub.extend(job_list)
            elif tokens and tokens[1] == 'Created':
                sub.extend(job_list)
            elif tokens and tokens[1] == 'CannotSubmit':
                match.extend(job_list)

        if resub or sub:
            jobs = ','.join(sorted(resub))
            mj = ','.join(sorted(match))
            script = """#!/usr/bin/env bash
crab -getoutput %s
crab -resubmit %s
#crab -forceResubmit %s
#created jobs not submitted
#crab -forceResubmit %s
#Cannot submit
#crab -match %s
#crab -resubmit %s
""" % (jobs,jobs,jobs, ','.join(sub),mj,mj)
            print script

    def __del__(self):

        if os.path.exists(self.xml_file):
            os.remove(self.xml_file)
                           

if __name__ == "__main__":

    c = CrabStatus()
    c.parseJobs()
    c.resubmit()
    c.printJobInfo()

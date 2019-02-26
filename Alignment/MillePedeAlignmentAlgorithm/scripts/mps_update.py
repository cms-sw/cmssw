#!/usr/bin/env python
from __future__ import print_function
import os
import re
import subprocess
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib

import six

def fill_time_info(mps_index, status, cpu_time):
    """Fill timing info in the database for `mps_index`.

    Arguments:
    - `mps_index`: index in the MPS database
    - `status`: job status
    - `cpu_time`: extracted CPU timing information
    """

    cpu_time = int(round(cpu_time))  # care only about seconds for now
    if status in ("RUN", "DONE"):
        if cpu_time > 0:
            diff = cpu_time - lib.JOBRUNTIME[mps_index]
            lib.JOBRUNTIME[mps_index] = cpu_time
            lib.JOBHOST[mps_index] = "+"+str(diff)
            lib.JOBINCR[mps_index] = diff
        else:
            lib.JOBRUNTIME[mps_index] = 0
            lib.JOBINCR[mps_index] = 0



################################################################################
# mapping of HTCondor status codes to MPS status
htcondor_jobstatus = {"1": "PEND", # Idle
                      "2": "RUN",  # Running
                      "3": "EXIT", # Removed
                      "4": "DONE", # Completed
                      "5": "PEND", # Held
                      "6": "RUN",  # Transferring output
                      "7": "PEND"} # Suspended


################################################################################
# collect submitted jobs (use 'in' to handle composites, e.g. DISABLEDFETCH)
lib = mpslib.jobdatabase()
lib.read_db()

submitted_jobs = {}
for i in xrange(len(lib.JOBID)):
    submitted = True
    for status in ("SETUP", "OK", "DONE", "FETCH", "ABEND", "WARN", "FAIL"):
        if status in lib.JOBSTATUS[i]:
            submitted = False
            break
    if submitted:
        submitted_jobs[lib.JOBID[i]] = i
print("submitted jobs:", len(submitted_jobs))


################################################################################
# deal with submitted jobs by looking into output of shell (bjobs/condor_q)
if len(submitted_jobs) > 0:
    job_status = {}
    if "htcondor" in lib.get_class("pede") or "htcondor" in lib.get_class("mille"):
        condor_q = subprocess.check_output(["condor_q", "-af:j",
                                            "JobStatus", "RemoteSysCpu"],
                                           stderr = subprocess.STDOUT)
        for line in condor_q.splitlines():
            job_id, status, cpu_time = line.split()
            job_status[job_id] = {"status": htcondor_jobstatus[status],
                                  "cpu": float(cpu_time)}

    bjobs = subprocess.check_output(["bjobs", "-l", "-a"],
                                    stderr = subprocess.STDOUT)
    bjobs = bjobs.replace("\n","")

    job_regex = re.compile(r"Job<(\d+?)>,")
    status_regex = re.compile(r"Status<([A-Z]+?)>")
    cputime_regex = re.compile(r"TheCPUtimeusedis(\d+(\.\d+)?)seconds")
    if bjobs != "No job found":
        results = bjobs.replace(" ","").split("-----------------------")
        for line in results:
            if len(line.strip()) == 0: continue
            # extract jobID
            job_id = job_regex.search(line).group(1)
            # extract job status
            status = status_regex.search(line).group(1)
            # extract CPU time (only present for finished job)
            match = cputime_regex.search(line)
            cpu_time = float(match.group(1)) if match else 0
            print("out ", job_id, " ", status, " ", cpu_time)
            job_status[job_id] = {"status": status,
                                  "cpu": cpu_time}

    for job_id, job_info in six.iteritems(job_status):
        mps_index = submitted_jobs.get(job_id, -1)
        # check for disabled Jobs
        disabled = "DISABLED" if "DISABLED" in lib.JOBSTATUS[mps_index] else ""

        # continue with next batch job if not found or not interesting
        if mps_index == -1:
            print("mps_update.py - the job", job_id, end=' ')
            print("was not found in the JOBID array")
            continue
        else:                   # pop entry from submitted jobs
            submitted_jobs.pop(job_id)


        # if found update Joblists for mps.db
        lib.JOBSTATUS[mps_index] = disabled+job_info["status"]
        fill_time_info(mps_index, job_info["status"], job_info["cpu"])


################################################################################
# loop over remaining jobs to see whether they are done
for job_id, mps_index in submitted_jobs.items(): # IMPORTANT to copy here (no iterator!)
    # check if current job is disabled. Print stuff.
    disabled = "DISABLED" if "DISABLED" in lib.JOBSTATUS[mps_index] else ""
    print(" DB job ", job_id, mps_index)

    # check if job may be done by looking if a folder exists in the project directory.
    # if True  -> jobstatus is set to DONE
    theBatchDirectory = "LSFJOB_"+job_id
    if os.path.isdir(theBatchDirectory):
        print("Directory ", theBatchDirectory, "exists")
        lib.JOBSTATUS[mps_index] = disabled + "DONE"
        submitted_jobs.pop(job_id)
        continue

    # check if it is a HTCondor job already moved to "history"
    elif "htcondor" in lib.get_class("pede") or "htcondor" in lib.get_class("mille"):
        userlog = os.path.join("jobData", lib.JOBDIR[mps_index], "HTCJOB")
        condor_h = subprocess.check_output(["condor_history", job_id, "-limit", "1",
                                            "-userlog", userlog,
                                            "-af:j", "JobStatus", "RemoteSysCpu"],
                                           stderr = subprocess.STDOUT)
        if len(condor_h.strip()) > 0:
            job_id, status, cpu_time = condor_h.split()
            status = htcondor_jobstatus[status]
            lib.JOBSTATUS[mps_index] = disabled + status
            fill_time_info(mps_index, status, float(cpu_time))
            submitted_jobs.pop(job_id)
            continue

    if "RUN" in lib.JOBSTATUS[mps_index]:
        print("WARNING: Job ", mps_index, end=' ')
        print("in state RUN, neither found by htcondor, nor bjobs, nor find", end=' ')
        print("LSFJOB directory!")


################################################################################
# check for orphaned jobs
for job_id, mps_index in six.iteritems(submitted_jobs):
    for status in ("SETUP", "DONE", "FETCH", "TIMEL", "SUBTD"):
        if status in lib.JOBSTATUS[mps_index]:
            print("Funny entry index", mps_index, " job", lib.JOBID[mps_index], end=' ')
            print(" status", lib.JOBSTATUS[mps_index])


lib.write_db()

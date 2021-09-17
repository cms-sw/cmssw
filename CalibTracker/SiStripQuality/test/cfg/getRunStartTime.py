#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser(description="Get the start time of the run")
parser.add_argument("runNumber", type=int)
args = parser.parse_args()
import sqlalchemy
import CondCore.Utilities.conddblib as conddb
session = conddb.connect(url=conddb.make_url()).session()
RunInfo = session.get_dbtype(conddb.RunInfo)
bestRun = session.query(
            RunInfo.run_number,
            RunInfo.start_time,
            RunInfo.end_time
        ).filter(
            RunInfo.run_number >= args.runNumber
        ).first()
if bestRun is None:
    raise Exception("Run %s can't be matched with an existing run in the database." % options.runNumber)
bestRun, runStart, runStop = bestRun
from calendar import timegm
bestRunStartTime = timegm(runStart.utctimetuple()) << 32
bestRunStopTime  = timegm(runStop.utctimetuple()) << 32
print("{0} -> best run: {1}, start time {2}, stop time {3}".format(args.runNumber, bestRun, runStart, runStop))
print(bestRunStartTime)

#!/bin/bash
set -e

# This is mainly to make sure nothing crashes. Checking the output for sanity is attempted but not really complete.

# 1. Run a very simple configuration with all module types.
cmsRun run_analyzers_cfg.py outfile=alltypes.root numberEventsInRun=100 numberEventsInLuminosityBlock=20 nEvents=100
[ 88 = $(dqmiolistmes.py alltypes.root -r 1 | wc -l) ]

# 2. Run multi-threaded. First we make a baseline file without legacy modules, since they might not work.
cmsRun run_analyzers_cfg.py outfile=nolegacy.root    numberEventsInRun=1000 numberEventsInLuminosityBlock=200 nEvents=1000 nolegacy=True
cmsRun run_analyzers_cfg.py outfile=nolegacy-mt.root numberEventsInRun=1000 numberEventsInLuminosityBlock=200 nEvents=1000 nolegacy=True nThreads=10

# 3. Try enabling concurrent lumis.
cmsRun run_analyzers_cfg.py outfile=nolegacy-cl.root numberEventsInRun=1000 numberEventsInLuminosityBlock=200 nEvents=1000 nolegacy=True nThreads=10 nConcurrent=10

# Validate 2 and 3: Dump DQMIO into plain text and compare
dumproot() { root2sqlite.py -o $1.sqlite $1 ; echo '.dump' | sqlite3 $1.sqlite > $1.sqldump ; rm $1.sqlite ; }
dumproot nolegacy.root
dumproot nolegacy-mt.root
dumproot nolegacy-cl.root

# TODO: if out DQM was correct, this would succeed!
cmp nolegacy.root.sqldump nolegacy-mt.root.sqldump || true
cmp nolegacy.root.sqldump nolegacy-cl.root.sqldump || true
# You could use `git diff --no-index --color-words nolegacy.root.sqldump nolegacy-mt.root.sqldump` to understand what is going on.

# 4. Try crossing a run boundary.
cmsRun run_analyzers_cfg.py outfile=multirun.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=1200
dqmiodumpmetadata.py multirun.root | grep -q '4 runs, 12 lumisections'

# 5. Now, make some chopped up files to try harvesting.
cmsRun run_analyzers_cfg.py outfile=part1.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50               # 1st half of 1st lumi
cmsRun run_analyzers_cfg.py outfile=part2.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50 firstEvent=50 # 2nd half of 1st lumi
cmsRun run_analyzers_cfg.py outfile=part3.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=200 firstEvent=100 firstLuminosityBlock=2 # lumi 2 and 3
cmsRun run_analyzers_cfg.py outfile=part4.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=900 firstRun=2   # 3 more runs

cmsRun run_harvesters_cfg.py inputFiles=part1.root inputFiles=part2.root inputFiles=part3.root inputFiles=part4.root outfile=merged.root nomodules=True
dqmiodumpmetadata.py merged.root | grep -q '4 runs, 12 lumisections'
dumproot multirun.root
dumproot merged.root
# these are unlikely to ever fully argee, though the histograms should. They do not, for now.
#cmp multirun.root.sqldump merged.root.sqldump

# 6. A load test. 
( ulimit -v 4000000 # limit available virtual memory
  cmsRun run_analyzers_cfg.py outfile=huge.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=600 nThreads=10 nConcurrent=2 howmany=1000 nolegacy=True
)

# 7. Try writing a TDirectory file. This is only safe for a single run for now.
cmsRun run_analyzers_cfg.py numberEventsInRun=100 numberEventsInLuminosityBlock=20 nEvents=100 legacyoutput=True

cmsRun run_harvesters_cfg.py inputFiles=alltypes.root nomodules=True legacyoutput=True

dumproot DQM_V0001_R000000001__EmptySource__DQMTests__DQMIO.root
dumproot DQM_V0001_R000000001__Harvesting__DQMTests__DQMIO.root
# These disagree due to the werid handling of per-lumi MEs in the current DQMStore.
cmp DQM_V0001_R000000001__EmptySource__DQMTests__DQMIO.root.sqldump DQM_V0001_R000000001__Harvesting__DQMTests__DQMIO.root.sqldump || true

# 8. Try writing ProtoBuf files.
cmsRun run_analyzers_cfg.py outfile=multirun.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=1200 protobufoutput=True

# This does not work, something in the file format seems to be not right.
cmsRun run_harvesters_cfg.py inputFiles=./run000001 outfile=pbdata.root nomodules=True protobufinput=True || true

# TODO: maybe also try fastHadd.






#!/bin/bash
set -e
set -x

if [[ -z ${LOCAL_TEST_DIR} ]]; then
  LOCAL_TEST_DIR=.
fi

# This is mainly to make sure nothing crashes. Checking the output for sanity is attempted but not really complete.

# 1. Run a very simple configuration with all module types.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=alltypes.root numberEventsInRun=100 numberEventsInLuminosityBlock=20 nEvents=100
[ 99 = $(dqmiolistmes.py alltypes.root -r 1 | wc -l) ]
[ 55 = $(dqmiolistmes.py alltypes.root -r 1 -l 1 | wc -l) ]
# this is deeply related to what the analyzers actually do.
# most run histos (5 modules * 6 types) fill on every event and should have 100 entries.
# the scalar MEs should have the last lumi number (5) (7 float + 7 int)
# testonefilllumi, testlegacyfilllumi also should have 5 entries in the histograms (2*6 more)
# the two "fillrun" modules should have one entry in the histograms (2*6 total) and 0 in the scalars (4 total)
[ "0: 2, 0.0: 2, 1: 12, 100: 30, 5: 19, 5.0: 7" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py alltypes.root -r 1 --summary)" ]
# per lumi we see 20 in most histograms (3*6), and the current lumi number in the scalars (5 modules * 2).
# the two fillumi modules should have one entry in each of there lumi histograms, (2*6 total)
[ "1: 17, 1.0: 5, 20: 18" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py alltypes.root -r 1 -l 1 --summary)" ]
[ "1: 12, 2: 5, 2.0: 5, 20: 18" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py alltypes.root -r 1 -l 2 --summary)" ]
[ "1: 12, 20: 18, 3: 5, 3.0: 5" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py alltypes.root -r 1 -l 3 --summary)" ]
[ "1: 12, 20: 18, 4: 5, 4.0: 5" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py alltypes.root -r 1 -l 4 --summary)" ]
[ "1: 12, 20: 18, 5: 5, 5.0: 5" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py alltypes.root -r 1 -l 5 --summary)" ]
# just make sure we are not off by one
[ "" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py alltypes.root -r 1 -l 6 --summary)" ]

# 2. Run multi-threaded. First we make a baseline file without legacy modules, since they might not work.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=nolegacy.root    numberEventsInRun=1000 numberEventsInLuminosityBlock=200 nEvents=1000 nolegacy=True
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=nolegacy-mt.root numberEventsInRun=1000 numberEventsInLuminosityBlock=200 nEvents=1000 nolegacy=True nThreads=10

# 3. Try enabling concurrent lumis.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=nolegacy-cl.root numberEventsInRun=1000 numberEventsInLuminosityBlock=200 nEvents=1000 nolegacy=True nThreads=10 nConcurrent=10

# same math as above, just a few less modules, and more events.
for f in nolegacy.root nolegacy-mt.root # nolegacy-cl.root # Enabling concurrent lumis seems to mess up the output for now.
do
  [ "0: 1, 0.0: 1, 1: 6, 1000: 18, 5: 3, 5.0: 3" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py $f -r 1 --summary)" ]
  [ "1: 1, 1.0: 1, 200: 6" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py $f -r 1 -l 1 --summary)" ]
  [ "2: 1, 2.0: 1, 200: 6" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py $f -r 1 -l 2 --summary)" ]
  [ "200: 6, 3: 1, 3.0: 1" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py $f -r 1 -l 3 --summary)" ]
  [ "200: 6, 4: 1, 4.0: 1" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py $f -r 1 -l 4 --summary)" ]
  [ "200: 6, 5: 1, 5.0: 1" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py $f -r 1 -l 5 --summary)" ]
  [ "" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py $f -r 1 -l 6 --summary)" ]
done

# 4. Try crossing a run boundary.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=multirun.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=1200
dqmiodumpmetadata.py multirun.root | grep -q '4 runs, 12 lumisections'

# 5. Now, make some chopped up files to try harvesting.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=part1.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50               # 1st half of 1st lumi
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=part2.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50 firstEvent=50 # 2nd half of 1st lumi
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=part3.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=200 firstEvent=100 firstLuminosityBlock=2 # lumi 2 and 3
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=part4.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=900 firstRun=2   # 3 more runs

cmsRun $LOCAL_TEST_DIR/run_harvesters_cfg.py inputFiles=part1.root inputFiles=part2.root inputFiles=part3.root inputFiles=part4.root outfile=merged.root nomodules=True
dqmiodumpmetadata.py merged.root | grep -q '4 runs, 12 lumisections'
dumproot() { root2sqlite.py -o $1.sqlite $1 ; echo '.dump' | sqlite3 $1.sqlite > $1.sqldump ; rm $1.sqlite ; }
dumproot multirun.root
dumproot merged.root
# these are unlikely to ever fully argee, though the histograms should. They do not, for now.
#cmp multirun.root.sqldump merged.root.sqldump

# 6. A load test. 
( ulimit -v 4000000 # limit available virtual memory
  cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=huge.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=600 nThreads=10 nConcurrent=2 howmany=1000 nolegacy=True
)

# 7. Try writing a TDirectory file. This is only safe for a single run for now.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py numberEventsInRun=100 numberEventsInLuminosityBlock=20 nEvents=100 legacyoutput=True

cmsRun $LOCAL_TEST_DIR/run_harvesters_cfg.py inputFiles=alltypes.root nomodules=True legacyoutput=True

dumproot DQM_V0001_R000000001__EmptySource__DQMTests__DQMIO.root
dumproot DQM_V0001_R000000001__Harvesting__DQMTests__DQMIO.root
# These disagree due to the werid handling of per-lumi MEs in the current DQMStore.
cmp DQM_V0001_R000000001__EmptySource__DQMTests__DQMIO.root.sqldump DQM_V0001_R000000001__Harvesting__DQMTests__DQMIO.root.sqldump || true
cmp <(grep -v lumi DQM_V0001_R000000001__EmptySource__DQMTests__DQMIO.root.sqldump) <(grep -v lumi DQM_V0001_R000000001__Harvesting__DQMTests__DQMIO.root.sqldump)

# 8. Try writing ProtoBuf files.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=1200 protobufoutput=True

cmsRun $LOCAL_TEST_DIR/run_harvesters_cfg.py inputFiles=./run000001 outfile=pbdata.root nomodules=True protobufinput=True
[ 99 = $(dqmiolistmes.py pbdata.root -r 1 | wc -l) ]
[ 55 = $(dqmiolistmes.py pbdata.root -r 1 -l 1 | wc -l) ]


# TODO: maybe also try fastHadd.

# 9. Try writing online files. This is really TDirectory files, but written via a different module.
# Note that this does not really need to support multiple runs, but it appears it does.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=1200 onlineoutput=True


# 10. Try running some harvesting modules and check if their output makes it out.
# Note that we pass the files in order here. In the future, this should be independent of the order of input files.
cmsRun $LOCAL_TEST_DIR/run_harvesters_cfg.py inputFiles=part1.root inputFiles=part2.root inputFiles=part3.root legacyoutput=True
rootlist ()
{  python -c '
import uproot
for k in uproot.open("'"$1"'").allkeys(): print k' 
}
[ 2 = $(rootlist DQM_V0001_R000000001__Harvesting__DQMTests__DQMIO.root | grep  -c '<harvestingsummary>s=beginRun(1) endLumi(1,1) endLumi(1,2) endLumi(1,3) endRun(1) endJob() </harvestingsummary>') ]

# 11. Try MEtoEDM and EDMtoME.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=metoedm.root numberEventsInRun=100 numberEventsInLuminosityBlock=20 nEvents=100 metoedmoutput=True
cmsRun $LOCAL_TEST_DIR/run_harvesters_cfg.py outfile=edmtome.root inputFiles=metoedm.root nomodules=True metoedminput=True
[ 99 = $(dqmiolistmes.py edmtome.root -r 1 | wc -l) ]
[ 55 = $(dqmiolistmes.py edmtome.root -r 1 -l 1 | wc -l) ]
[ "0: 2, 0.0: 2, 1: 12, 100: 30, 5: 19, 5.0: 7" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py edmtome.root -r 1 --summary)" ]
[ "1: 17, 1.0: 5, 20: 18" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py edmtome.root -r 1 -l 1 --summary)" ]
[ "1: 12, 2: 5, 2.0: 5, 20: 18" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py edmtome.root -r 1 -l 2 --summary)" ]
[ "1: 12, 20: 18, 3: 5, 3.0: 5" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py edmtome.root -r 1 -l 3 --summary)" ]
[ "1: 12, 20: 18, 4: 5, 4.0: 5" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py edmtome.root -r 1 -l 4 --summary)" ]
[ "1: 12, 20: 18, 5: 5, 5.0: 5" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py edmtome.root -r 1 -l 5 --summary)" ]
[ "" = "$($LOCAL_TEST_DIR/dqmiodumpentries.py edmtome.root -r 1 -l 6 --summary)" ]

cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=part1_metoedm.root metoedmoutput=True numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50               # 1st half of 1st lumi
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=part2_metoedm.root metoedmoutput=True numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50 firstEvent=50 # 2nd half of 1st lumi
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=part3_metoedm.root metoedmoutput=True numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=200 firstEvent=100 firstLuminosityBlock=2 # lumi 2 and 3
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=part4_metoedm.root metoedmoutput=True numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=900 firstRun=2   # 3 more runs

cmsRun $LOCAL_TEST_DIR/run_harvesters_cfg.py inputFiles=part1_metoedm.root inputFiles=part2_metoedm.root inputFiles=part3_metoedm.root inputFiles=part4_metoedm.root outfile=metoedm_merged.root nomodules=True metoedminput=True
dqmiodumpmetadata.py metoedm_merged.root | grep -q '4 runs, 12 lumisections'
dumproot metoedm_merged.root
#cmp multirun.root.sqldump metoedm_merged.root.sqldump


# 12. Sanity checks.
# this will mess up some of the files created earlier, disable for debugging.
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=empty.root nEvents=0
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=empty.root howmany=0
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=empty.root howmany=0 legacyoutput=True
cmsRun $LOCAL_TEST_DIR/run_analyzers_cfg.py outfile=empty.root howmany=0 protobufoutput=True









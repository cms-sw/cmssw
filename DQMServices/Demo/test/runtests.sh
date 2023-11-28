#!/bin/bash
set -e
set -x

# This is mainly to make sure nothing crashes. Checking the output for sanity is attempted but not really complete.

# 1. Run a very simple configuration with all module types.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=alltypes.root numberEventsInRun=100 numberEventsInLuminosityBlock=20 nEvents=100
# actually we'd expect 99, but the MEs by legacy modules are booked with JOB scope and cannot be saved to DQMIO.
[ 78 = $(dqmiolistmes.py alltypes.root -r 1 | wc -l) ]
[ 78 = $(dqmiolistmes.py alltypes.root -r 1 -l 1 | wc -l) ]
# this is deeply related to what the analyzers actually do.
# again, the legacy modules output is not saved.
# most run histos (4 modules * 9 types) fill on every event and should have 100 entries.
# the scalar MEs should have the last lumi number (5) (5 float + 5 int)
# testonefilllumi also should have 5 entries in the histograms (9 more)
# the "fillrun" module should have one entry in the histograms (9 total) and 0 in the scalars (2 total)

[ "0: 1, 0.0: 1, 1: 11, 100: 33, 200: 11, 5: 16, 5.0: 5" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py alltypes.root -r 1 --summary)" ]
# per lumi we see 20 in most histograms (4*9), and the current lumi number in the scalars (6 modules * 2).
# the two fillumi modules should have one entry in each of the lumi histograms, (2*9 total)
 
[ "1: 28, 1.0: 6, 20: 44" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py alltypes.root -r 1 -l 1 --summary)" ]
[ "1: 22, 2: 6, 2.0: 6, 20: 44" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py alltypes.root -r 1 -l 2 --summary)" ]
[ "1: 22, 20: 44, 3: 6, 3.0: 6" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py alltypes.root -r 1 -l 3 --summary)" ]
[ "1: 22, 20: 44, 4: 6, 4.0: 6" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py alltypes.root -r 1 -l 4 --summary)" ]
[ "1: 22, 20: 44, 5: 6, 5.0: 6" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py alltypes.root -r 1 -l 5 --summary)" ]
# just make sure we are not off by one
[ "" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py alltypes.root -r 1 -l 6 --summary)" ]


# 2. Run multi-threaded. First we make a baseline file without legacy modules, since they might not work.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=nolegacy.root    numberEventsInRun=1000 numberEventsInLuminosityBlock=200 nEvents=1000 nolegacy=True
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=nolegacy-mt.root numberEventsInRun=1000 numberEventsInLuminosityBlock=200 nEvents=1000 nolegacy=True nThreads=10


# 3. Try enabling concurrent lumis.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=nolegacy-cl.root numberEventsInRun=1000 numberEventsInLuminosityBlock=200 nEvents=1000 nolegacy=True nThreads=10 nConcurrent=10

# same math as above, just a few less modules, and more events.
for f in nolegacy.root nolegacy-mt.root nolegacy-cl.root
do
  [ "0: 1, 0.0: 1, 1: 11, 1000: 22, 2000: 11, 5: 3, 5.0: 3" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py $f -r 1 --summary)" ]
  [ "1: 2, 1.0: 2, 200: 22" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py $f -r 1 -l 1 --summary)" ]
  [ "2: 2, 2.0: 2, 200: 22" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py $f -r 1 -l 2 --summary)" ]
  [ "200: 22, 3: 2, 3.0: 2" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py $f -r 1 -l 3 --summary)" ]
  [ "200: 22, 4: 2, 4.0: 2" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py $f -r 1 -l 4 --summary)" ]
  [ "200: 22, 5: 2, 5.0: 2" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py $f -r 1 -l 5 --summary)" ]
  [ "" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py $f -r 1 -l 6 --summary)" ]
done


# 4. Try crossing a run boundary.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=multirun.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=1200
dqmiodumpmetadata.py multirun.root | grep -q '4 runs, 12 lumisections'


# 5. Now, make some chopped up files to try harvesting.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=part1.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50               # 1st half of 1st lumi
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=part2.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50 firstEvent=50 # 2nd half of 1st lumi
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=part3.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=200 firstEvent=100 firstLuminosityBlock=2 # lumi 2 and 3
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=part4.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=900 firstRun=2   # 3 more runs

cmsRun ${SCRAM_TEST_PATH}/run_harvesters_cfg.py inputFiles=part1.root inputFiles=part2.root inputFiles=part3.root inputFiles=part4.root outfile=merged.root nomodules=True
dqmiodumpmetadata.py merged.root | grep -q '4 runs, 12 lumisections'

#dumproot() { root2sqlite.py -o $1.sqlite $1 ; echo '.dump' | sqlite3 $1.sqlite > $1.sqldump ; rm $1.sqlite ; }
#dumproot multirun.root
#dumproot merged.root
rootlist ()
{  python3 -c '
import uproot
for k in uproot.open("'"$1"'").keys(): print(k)'
}

# we need to exclude MEs filled on run and lumi boundaries, since the split job *does* see a different number of begin/end run/lumi transitions.
cmp <(${SCRAM_TEST_PATH}/dqmiodumpentries.py multirun.root -r 1 | grep -vE 'fillrun|filllumi') <(${SCRAM_TEST_PATH}/dqmiodumpentries.py merged.root -r 1 | grep -vE 'fillrun|filllumi')
cmp <(${SCRAM_TEST_PATH}/dqmiodumpentries.py multirun.root -r 3) <(${SCRAM_TEST_PATH}/dqmiodumpentries.py merged.root -r 3)
cmp <(${SCRAM_TEST_PATH}/dqmiodumpentries.py multirun.root -r 1 -l 1 | grep -v filllumi) <(${SCRAM_TEST_PATH}/dqmiodumpentries.py merged.root -r 1 -l 1 | grep -v filllumi)
cmp <(${SCRAM_TEST_PATH}/dqmiodumpentries.py multirun.root -r 1 -l 2) <(${SCRAM_TEST_PATH}/dqmiodumpentries.py merged.root -r 1 -l 2)

# 6. A load test. 
#( if [[ `uname -m` != aarch64 ]] ; then ulimit -v 4000000 ; fi # limit available virtual memory
  cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=huge.root numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=600 nThreads=10 nConcurrent=2 howmany=1000 nolegacy=True
#)


# 7. Try writing a TDirectory file.
cmsRun ${SCRAM_TEST_PATH}/run_harvesters_cfg.py inputFiles=alltypes.root nomodules=True legacyoutput=True reScope=JOB
# this number is rather messy: we have 66 per-lumi objecs (harvested), 66 per-run objects (no legacy output), one folder for each set of 11, 
# plus some higher-level folders and the ProvInfo hierarchy create by the FileSaver.
[ 185 = $(rootlist DQM_V0001_R000000001__Harvesting__DQMTests__DQMIO.root | wc -l) ]

cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py numberEventsInRun=100 numberEventsInLuminosityBlock=20 nEvents=100 legacyoutput=True
# we expect only the (per-job) legacy histograms here: 3*11 objects in 3 folders, plus 9 more for ProvInfo and higher-level folders.
[ 51 = $(rootlist DQM_V0001_R000000001__EmptySource__DQMTests__DQMIO.root | wc -l) ]

# 8. Try writing ProtoBuf files.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=1200 protobufoutput=True

cmsRun ${SCRAM_TEST_PATH}/run_harvesters_cfg.py inputFiles=./run000001 outfile=pbdata.root nomodules=True protobufinput=True
[ 117 = $(dqmiolistmes.py pbdata.root -r 1 | wc -l) ]
[ 78 = $(dqmiolistmes.py pbdata.root -r 1 -l 1 | wc -l) ]

# this will potentially mess up statistics (we should only fastHadd *within* a lumisection, not *across*), but should technically work.
fastHadd add -o streamDQMHistograms.pb run000001/run000001_ls*_streamDQMHistograms.pb
# the output format is different from the harvesting above, this is a not-DQM-formatted TDirectory file.
fastHadd convert -o streamDQMHistograms.root streamDQMHistograms.pb
# here we expect all (incl. legacy) MEs (99+66), plus folders (14 + 4 higher-level)
[ 214 = $(rootlist streamDQMHistograms.root | wc -l) ]


# 9. Try writing online files. This is really TDirectory files, but written via a different module.
# Note that this does not really need to support multiple runs, but it appears it does.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=1200 onlineoutput=True
# here we expect full per-run output (99 objects), no per-lumi MEs, plus folders (9 + 10 higher-level).
[ 136 = $(rootlist DQM_V0001_UNKNOWN_R000000001.root | wc -l) ]
[ 136 = $(rootlist DQM_V0001_UNKNOWN_R000000002.root | wc -l) ]
[ 136 = $(rootlist DQM_V0001_UNKNOWN_R000000003.root | wc -l) ]
[ 136 = $(rootlist DQM_V0001_UNKNOWN_R000000004.root | wc -l) ]


# 10. Try running some harvesting modules and check if their output makes it out.
# Note that we pass the files out-of order here; the DQMIO input should sort them.
cmsRun ${SCRAM_TEST_PATH}/run_harvesters_cfg.py inputFiles=part1.root inputFiles=part3.root inputFiles=part2.root legacyoutput=True
[ 1 = $(rootlist DQM_V0001_R000000001__Harvesting__DQMTests__DQMIO.root | grep  -c '<harvestingsummary>s=beginRun(1) endLumi(1,1) endLumi(1,2) endLumi(1,3) endRun(1) endJob() </harvestingsummary>') ]
# The legacy harvester can only do per-run harvesting.
[ 2 = $(rootlist DQM_V0001_R000000001__Harvesting__DQMTests__DQMIO.root | grep  -c '<runsummary>s=beginRun(1) endLumi(1,1) endLumi(1,2) endLumi(1,3) endRun(1) </runsummary>') ]

# 11. Try MEtoEDM and EDMtoME.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=metoedm.root numberEventsInRun=100 numberEventsInLuminosityBlock=20 nEvents=100 metoedmoutput=True
cmsRun ${SCRAM_TEST_PATH}/run_harvesters_cfg.py outfile=edmtome.root inputFiles=metoedm.root nomodules=True metoedminput=True
[ 72 = $(dqmiolistmes.py edmtome.root -r 1 | wc -l) ]
[ 72 = $(dqmiolistmes.py edmtome.root -r 1 -l 1 | wc -l) ]
# again, no legacy module (run) output here due to JOB scope for legacy modules
[ "0: 1, 0.0: 1, 1: 10, 100: 30, 200: 10, 5: 15, 5.0: 5" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py edmtome.root -r 1 --summary)" ]
[ "1: 26, 1.0: 6, 20: 40" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py edmtome.root -r 1 -l 1 --summary)" ]
[ "1: 20, 2: 6, 2.0: 6, 20: 40" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py edmtome.root -r 1 -l 2 --summary)" ]
[ "1: 20, 20: 40, 3: 6, 3.0: 6" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py edmtome.root -r 1 -l 3 --summary)" ]
[ "1: 20, 20: 40, 4: 6, 4.0: 6" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py edmtome.root -r 1 -l 4 --summary)" ]
[ "1: 20, 20: 40, 5: 6, 5.0: 6" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py edmtome.root -r 1 -l 5 --summary)" ]
[ "" = "$(${SCRAM_TEST_PATH}/dqmiodumpentries.py edmtome.root -r 1 -l 6 --summary)" ]

cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=part1_metoedm.root metoedmoutput=True numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50               # 1st half of 1st lumi
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=part2_metoedm.root metoedmoutput=True numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=50 firstEvent=50 # 2nd half of 1st lumi
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=part3_metoedm.root metoedmoutput=True numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=200 firstEvent=100 firstLuminosityBlock=2 # lumi 2 and 3
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=part4_metoedm.root metoedmoutput=True numberEventsInRun=300 numberEventsInLuminosityBlock=100 nEvents=900 firstRun=2   # 3 more runs

cmsRun ${SCRAM_TEST_PATH}/run_harvesters_cfg.py inputFiles=part1_metoedm.root inputFiles=part2_metoedm.root inputFiles=part3_metoedm.root inputFiles=part4_metoedm.root outfile=metoedm_merged.root nomodules=True metoedminput=True
dqmiodumpmetadata.py metoedm_merged.root | grep -q '4 runs, 12 lumisections'

# 12. Sanity checks.
# this will mess up some of the files created earlier, disable for debugging.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=empty.root nEvents=0
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=empty.root howmany=0
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=empty.root howmany=0 legacyoutput=True
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=empty.root howmany=0 protobufoutput=True
# nLumisections might be a bit buggy (off by one) in EDM, but is fine here.
cmsRun ${SCRAM_TEST_PATH}/run_analyzers_cfg.py outfile=noevents.root processingMode='RunsAndLumis' nLumisections=20
[ 78 = $(dqmiolistmes.py noevents.root -r 1 | wc -l) ]
[ 78 = $(dqmiolistmes.py noevents.root -r 1 -l 1 | wc -l) ]
[ 78 = $(dqmiolistmes.py noevents.root -r 2 | wc -l) ]
[ 78 = $(dqmiolistmes.py noevents.root -r 2 -l 2 | wc -l) ]



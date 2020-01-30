#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/OfflineValidation ..."
cmsRun ${LOCAL_TEST_DIR}/test_all_cfg.py || die "Failure running test_OfflineValidation_cfg.py" $?

if test -f "validation_config.ini"; then
    rm -f validation_config.ini
fi

cat <<EOF >> validation_config.ini
[general]
jobmode = interactive
eosdir  = Test

[alignment:prompt]
title = prompt
globaltag = 92X_dataRun2_Prompt_v2
color = 1
style = 2001

[alignment:express]
title = express
globaltag = 92X_dataRun2_Express_v2
color = 2
style = 2402

[offline:validation_MinBias]
multiIOV  = false
maxevents = 10
dataset   = /MinimumBias/Run2017A-TkAlMinBias-PromptReco-v1/ALCARECO
magneticfield = 3.8
trackcollection = ALCARECOTkAlMinBias

[offline:validation_cosmics]
multiIOV  = false
maxevents = 10
dataset = /Cosmics/Run2017A-TkAlCosmics0T-PromptReco-v1/ALCARECO
magneticfield = 3.8
trackcollection = ALCARECOTkAlCosmicsCTF0T

[compare:Tracker]
multiIOV = false
levels = "Tracker","DetUnit"
dbOutput = false

[zmumu:some_zmumu_validation]
multiIOV = false
maxevents = 10
dataset = /DoubleMuon/Run2017A-TkAlZMuMu-PromptReco-v3/ALCARECO
etamaxneg = 2.4
etaminneg = -2.4
etamaxpos = 2.4
etaminpos = -2.4

[split:some_split_validation]
multiIOV = false
maxevents = 10
dataset = /Cosmics/Run2017A-TkAlCosmics0T-PromptReco-v1/ALCARECO
trackcollection = ALCARECOTkAlCosmicsCTF0T

[plots:offline]
DMROptions = plain split
DMRMinimum = 5
legendoptions = meanerror rmserror modules outside
customtitle = #CMS{Preliminary}
customrighttitle = 2017A cosmics and collisions data
legendheader = header
bigtext = true

[plots:split]
outliercut = 0.95

customtitle = #CMS{Preliminary}
customrighttitle = 2017A 3.8T cosmics data
legendheader = header

[plots:zmumu]
customtitle = #CMS{Preliminary}
customrighttitle = 2016G Z#rightarrow#mu#mu data, |#eta|<2.4
legendheader = header

[validation]
offline validation_MinBias - prompt :
offline validation_MinBias - express :
offline validation_cosmics - prompt :
offline validation_cosmics - express :
compare Tracker - prompt 278819, express 278819 :
zmumu some_zmumu_validation - prompt :
zmumu some_zmumu_validation - express :
split some_split_validation - prompt :
split some_split_validation - express :
EOF

validateAlignments.py -c validation_config.ini -N testingAllInOneTool --dryRun || die "Failure running all-in-one test" $?

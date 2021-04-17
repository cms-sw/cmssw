#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/OfflineValidation ..."
cmsRun ${LOCAL_TEST_DIR}/test_all_cfg.py || die "Failure running test_OfflineValidation_cfg.py" $?

if test -f "validation_config.ini"; then
    rm -f validation_config.ini
fi

## copy into local sqlite file the ideal alignment
echo "COPYING locally Ideal Alignment ..."
conddb --yes --db pro copy TrackerAlignment_Upgrade2017_design_v4 --destdb myfile.db
conddb --yes --db pro copy TrackerAlignmentErrorsExtended_Upgrade2017_design_v0 --destdb myfile.db

echo "GENERATING all-in-one tool configuration ..."
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
condition TrackerAlignmentRcd =  sqlite_file:myfile.db,TrackerAlignment_Upgrade2017_design_v4
condition TrackerAlignmentErrorExtendedRcd = sqlite_file:myfile.db,TrackerAlignmentErrorsExtended_Upgrade2017_design_v0
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

[primaryvertex:validation_HLTPhysics]
maxevents = 10000
multiIOV = false
dataset = /HLTPhysics/Run2017A-TkAlMinBias-PromptReco-v1/ALCARECO
trackcollection = ALCARECOTkAlMinBias
vertexcollection = offlinePrimaryVertices
isda = True
ismc = True
numberOfBins = 48
runboundary = 1
lumilist = None
ptCut  = 3.
etaCut = 2.5
runControl = False

[pvresolution:validation_JetHT]
multiIOV = false
maxevents = 50000
dataset = /JetHT/Run2017B-TkAlMinBias-09Aug2019_UL2017-v1/ALCARECO
trackcollection = ALCARECOTkAlMinBias
runboundary = 1
runControl = False
doTriggerSelection = False
triggerBits = "*"

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

[plots:primaryvertex]
doMaps = true
stdResiduals = true
autoLimits = false
m_dxyPhiMax = 40
m_dzPhiMax = 40
m_dxyEtaMax = 40
m_dzEtaMax = 40
m_dxyPhiNormMax = 0.5
m_dzPhiNormMax = 0.5
m_dxyEtaNormMax = 0.5
m_dzEtaNormMax = 0.5
w_dxyPhiMax = 150
w_dzPhiMax = 150
w_dxyEtaMax = 150
w_dzEtaMax = 1000
w_dxyPhiNormMax = 1.8
w_dzPhiNormMax = 1.8
w_dxyEtaNormMax = 1.8
w_dzEtaNormMax = 1.8

[validation]
offline validation_MinBias - prompt :
offline validation_MinBias - express :
offline validation_cosmics - prompt :
offline validation_cosmics - express :
primaryvertex validation_HLTPhysics - prompt :
primaryvertex validation_HLTPhysics - express :
pvresolution validation_JetHT - prompt :
pvresolution validation_JetHT - express :
compare Tracker - prompt 278819, express 278819 :
zmumu some_zmumu_validation - prompt :
zmumu some_zmumu_validation - express :
split some_split_validation - prompt :
split some_split_validation - express :
EOF

echo " TESTING all-in-one tool ..."
validateAlignments.py -c validation_config.ini -N testingAllInOneTool --dryRun || die "Failure running all-in-one test" $?

printf "\n\n"

echo " TESTING Primary Vertex Validation run-by-run submission ..."
submitPVValidationJobs.py -j UNIT_TEST -D /HLTPhysics/Run2016C-TkAlMinBias-07Dec2018-v1/ALCARECO -i ${LOCAL_TEST_DIR}/testPVValidation_Relvals_DATA.ini -r --unitTest || die "Failure running PV Validation run-by-run submission" $?

printf "\n\n"

echo " TESTING Split Vertex Validation submission ..."
submitPVResolutionJobs.py -j UNIT_TEST -D /JetHT/Run2018C-TkAlMinBias-12Nov2019_UL2018-v2/ALCARECO -i ${LOCAL_TEST_DIR}/PVResolutionExample.ini --unitTest || die "Failure running Split Vertex Validation submission" $?

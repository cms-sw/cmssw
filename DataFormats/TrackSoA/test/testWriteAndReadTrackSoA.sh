echo '#### Test Writing and Reading TrackSoA'

scriptdir=$CMSSW_BASE/src/DataFormats/TrackSoA/test/

echo '> Writing'

cmsRun ${scriptdir}/testWriteHostTrackSoA.py testTrackSoa.root

if [ $? -ne 0 ]; then
    exit 1;
fi

echo '> Reading'

cmsRun ${scriptdir}/testReadHostTrackSoA.py testTrackSoa.root

if [ $? -ne 0 ]; then
    exit 1;
fi

echo '>>>> Done! <<<<'


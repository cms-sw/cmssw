echo '#### Test Writing and Reading TrackingRecHitSoA'

scriptdir=$CMSSW_BASE/src/DataFormats/TrackingRecHitSoA/test/

echo '> Writing'

cmsRun ${scriptdir}/testWriteHostHitSoA.py testHitSoa.root

if [ $? -ne 0 ]; then
   exit 1;
fi

echo '> Reading'

cmsRun ${scriptdir}/testReadHostHitSoA.py testHitSoa.root

if [ $? -ne 0 ]; then
   exit 1;
fi

echo '>>>> Done! <<<<'


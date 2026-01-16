echo '#### Test Writing and Reading TrackSoA'

scriptdir=$CMSSW_BASE/src/DataFormats/VertexSoA/test/

echo '> Writing'

cmsRun ${scriptdir}/testWriteHostVertexSoA.py testVertexSoA.root

if [ $? -ne 0 ]; then
    exit 1;
fi

echo '> Reading'

cmsRun ${scriptdir}/testReadHostVertexSoA.py testVertexSoA.root

if [ $? -ne 0 ]; then
    exit 1;
fi

echo '>>>> Done! <<<<'


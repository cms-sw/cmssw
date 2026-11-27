echo '#### Test Writing and Reading VertexGNNSoA'

scriptdir=$CMSSW_BASE/src/DataFormats/VertexSoA/test/

echo '> Writing'

cmsRun ${scriptdir}/testWriteHostVertexGNNSoA.py testVertexGNNSoA.root

if [ $? -ne 0 ]; then
    exit 1;
fi

echo '> Reading'

cmsRun ${scriptdir}/testReadHostVertexGNNSoA.py testVertexGNNSoA.root

if [ $? -ne 0 ]; then
    exit 1;
fi

echo '>>>> Done! <<<<'

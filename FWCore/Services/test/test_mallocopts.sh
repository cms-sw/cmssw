
TMP_CFG="/tmp/test${$}.cfg"
ANSWER="Malloc options: mmap_max=100001 trim_threshold=100002 top_padding=100003 mmap_threshold=100004"

cat > ${TMP_CFG}  <<EOF
process Sim  = 
{
  untracked PSet maxEvents = { untracked int32 input = 2 }
  source = EmptySource { }
  module thing = ThingProducer { }
  path p = { thing }

  service = InitRootHandlers { untracked bool UnloadRootSigHandler = true }

  service = SimpleMemoryCheck
  {
    untracked int32 M_MMAP_MAX = 100001
    untracked int32 M_TRIM_THRESHOLD = 100002
    untracked int32 M_TOP_PAD = 100003
    untracked int32 M_MMAP_THRESHOLD = 100004
    untracked bool dump = true
  }
}
EOF

RESULT=`cmsRun ${TMP_CFG} 2>&1 | grep "Malloc opt"`

if [ "$ANSWER" != "$RESULT" ]
then
	echo "Failed"
	exit 1
fi

rm -f ${TMP_CFG}
exit 0

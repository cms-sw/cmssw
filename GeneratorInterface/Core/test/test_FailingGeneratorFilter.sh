# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }


echo "******************"
echo "Signal in constructor"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 0 1) && die "signal in constructor didn't fail" 1
echo "******************"
echo "Signal in beginLumi"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 1 1) && die "signal in begin lumi didn't fail" 1
echo "******************"
echo "Signal in event"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 2 1) && die "signal in event didn't fail" 1


echo "******************"
echo "Exception in constructor"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 0 0) && die "exception in constructor didn't fail" 1
echo "******************"
echo "Exception in beginLumi"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 1 0) && die "exception in begin lumi didn't fail" 1
echo "******************"
echo "Exception in event"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 2 0) && die "exception in event didn't fail" 1

echo "******************"
echo "terminate in constructor"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 0 2) && die "signal in constructor didn't fail" 1
echo "******************"
echo "terminate in beginLumi"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 1 2) && die "signal in begin lumi didn't fail" 1
echo "******************"
echo "terminate in event"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 2 2) && die "signal in event didn't fail" 1

echo "******************"
echo "exit in constructor"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 0 3) && die "signal in constructor didn't fail" 1
echo "******************"
echo "exit in beginLumi"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 1 3) && die "signal in begin lumi didn't fail" 1
echo "******************"
echo "exit in event"
(cmsRun ${LOCALTOP}/src/GeneratorInterface/Core/test/test_FailingGeneratorFilter_cfg.py 2 3) && die "signal in event didn't fail" 1


exit 0

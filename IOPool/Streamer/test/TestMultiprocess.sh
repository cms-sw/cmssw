function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/streamer_multiprocess_gen_file_cfg.py || die 'Failure using streamer_multiprocess_gen_file_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/streamer_multiprocess_cfg.py || die 'Failure using streamer_multiprocess_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/streamer_multiprocess_gen_file_oneLumi_cfg.py || die 'Failure using streamer_multiprocess_gen_file_oneLumi_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/streamer_multiprocess_WithSkip_cfg.py || die 'Failure using streamer_multiprocess_gen_file_WithSkip_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/streamer_multiprocess_oneLumi_cfg.py || die 'Failure using streamer_multiprocess_oneLumi_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/streamer_multiprocess_gen_file_oneLumi2_cfg.py || die 'Failure using streamer_multiprocess_gen_file_oneLumi2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/streamer_multiprocess_multiRun_cfg.py || die 'Failure using streamer_multiprocess_multiRun_cfg.py' $?

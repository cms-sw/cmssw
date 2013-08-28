#### Source this file

#cmsRun muonPostProcessor_cfg.py print files='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_zmm_2_5_all.root' output='file:/data/ndpc1/b/slaunwhj/scratch0/Histos_zmm_2_5_all.root' maxEvents=-1 >&! ~/scratch0/test_zmm_post.log
#cmsRun muonPostProcessor_cfg.py print files='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_ttbar_2_5_all.root' output='file:/data/ndpc1/b/slaunwhj/scratch0/Histos_ttbar_2_5_all.root' maxEvents=-1 > & ! ~/scratch0/test_ttbar_post.log 
#cmsRun muonPostProcessor_cfg.py print files='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_singleMu10_2_5_all.root' output='file:/data/ndpc1/b/slaunwhj/scratch0/Histos_singleMu10_2_5_all.root' maxEvents=-1 > & ! ~/scratch0/test_singleMu_post.log 

#cmsRun muonPostProcessor_cfg.py print files='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_craftSuperPointing.root' outputDir='/data/ndpc1/b/slaunwhj/scratch0/' maxEvents=-1 > & ! ~/scratch0/test_cosmics_post.log 
#cmsRun muonClientTest_cfg.py print files='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_ttbar_2_5_all.root' outputDir='/data/ndpc1/b/slaunwhj/scratch0/' workflow='/DQMGeneric/Test/Blah' maxEvents=-1 >&! ~/scratch0/test_top_post.log

#cmsRun muonPostProcessor_cfg.py print files='file:/data/ndpc0/b/slaunwhj/scratch0/EDM_jpsi_330_pre3_fullTagProbe.root' outputDir='/data/ndpc0/b/slaunwhj/scratch0/' workflow='/DQMGeneric/JPsi/34X' maxEvents=-1 >&! /data/ndpc0/b/slaunwhj/scratch0/post.log
#cmsRun muonPostProcessor_cfg.py print files='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_qcdFlatPt_fullTnP.root' outputDir='/data/ndpc0/b/slaunwhj/scratch0/' workflow='/DQMGeneric/QCD_FlatPt/34X' maxEvents=-1 >&! /data/ndpc0/b/slaunwhj/scratch0/post.log
#cmsRun muonPostProcessor_cfg.py print files='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_minbias_fullTnP.root' outputDir='/data/ndpc0/b/slaunwhj/scratch0/' workflow='/DQMGeneric/MinBias/34X' maxEvents=-1 >&! /data/ndpc0/b/slaunwhj/scratch0/post.log

cmsRun muonPostProcessor_cfg.py print files='file:/data/ndpc0/b/slaunwhj/scratch0/EDM_zmm_340pre4_numEvent100.root' outputDir='/data/ndpc0/b/slaunwhj/scratch0/' workflow='/DQMGeneric/PostProcess/Testv1' maxEvents=-1 >&! /data/ndpc0/b/slaunwhj/scratch0/post.log

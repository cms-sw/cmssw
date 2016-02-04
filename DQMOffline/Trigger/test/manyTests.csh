#### Source this file


#cmsRun muonTest_cfg.py print files_load=zmm_3_2_5.list output='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_zmm_2_5_all.root' maxEvents=-1 > & ! ~/scratch0/test_zmm.log 

#cmsRun muonTest_cfg.py print files_load=singleMu10.list output='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_singleMu10_2_5_all.root' maxEvents=-1 > & ! ~/scratch0/test_singleMu.log 

#cmsRun muonTest_cfg.py print files_load=jpsi.list output='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_jpsi_all.root' maxEvents=-1 > & ! ~/scratch0/test_jpsi.log 

#cmsRun muonTest_cfg.py print files_load=ttbar_3_1_0_pre9.list output='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_ttbar_2_5_all.root' maxEvents=-1 > & ! ~/scratch0/test_ttbar.log 



#cmsRun quadJetTest_cfg.py print files_load=ttbar_3_1_0_pre9.list output='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_ttbar_qjet_test.root' maxEvents=10 > & ! ~/scratch0/test_qjet.log

#cmsRun  muonTest_cfg.py print files_load=qcd_flatpt.list output='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_qcdFlatPt_fullTnP.root' maxEvents=-1 > & ! /data/ndpc0/b/slaunwhj/scratch0/test_fullTNP.log
#cmsRun  muonTest_cfg.py print files_load=minbias.list output='file:/data/ndpc1/b/slaunwhj/scratch0/EDM_minbias_fullTnP.root' maxEvents=-1 > & ! /data/ndpc0/b/slaunwhj/scratch0/test_fullTNP.log


######  ZMM
#cmsRun muonTest_cfg.py print files_load=zmm_340_castor_one.list output='file:/data/ndpc0/b/slaunwhj/scratch0/EDM_zmm_fullTNP.root' maxEvents=-1 > & ! /data/ndpc0/b/slaunwhj/scratch0/test_fullTNP.log

######  Upsilon
#cmsRun muonTest_cfg.py print files_load=upsilon_330.list output='file:/data/ndpc0/b/slaunwhj/scratch0/EDM_upsilon_fullTNP.root' maxEvents=-1 > & ! /data/ndpc0/b/slaunwhj/scratch0/test_fullTNP_ups.log

###### JPSI
#cmsRun muonTest_cfg.py print files_load=jpsi_330_pre3.list output='file:/data/ndpc0/b/slaunwhj/scratch0/EDM_jpsi_330_pre3_fullTagProbe.root' maxEvents=-1 > & ! /data/ndpc0/b/slaunwhj/scratch0/test_jpsi.log

#cmsRun quadJetTest_cfg.py print files_load=tt_oct.list output='file:/data/ndpc0/b/slaunwhj/scratch0/EDM_ttbar_qjet_test.root' maxEvents=500 > & ! /data/ndpc0/b/slaunwhj/scratch0/test_qjet.log

cmsRun muonTest_cfg.py print files_load=zmm_340_pre4.list output='file:/data/ndpc0/b/slaunwhj/scratch0/EDM_zmm_340pre4.root' maxEvents=100 > & ! /data/ndpc0/b/slaunwhj/scratch0/test_340pre4.log

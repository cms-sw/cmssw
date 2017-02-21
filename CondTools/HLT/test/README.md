Commands to run the workflow multi-IOV:

## TO UPDATE
```
python run_wf.py -f frontier://PromptProd/CMS_CONDITIONS -i AlCaRecoHLTpaths8e29_1e31_v24_offline -d AlCaRecoHLTpaths_TEST
```

will create an update sqlite file called `AlCaRecoHLTpaths_TEST.db` with an updated tag ` AlCaRecoHLTpaths_TEST` (the same IOV structure will be preserved)

Options available:
   * `-f` specifies connection (allows both condDB and sqlite files)
   * `-i` specifies input tag
   * `-d` specifies the output tag
   * `[-C]` (optional) if set to false leaves transient files IOV-by-IOV


## TO READ BACK
```
cmsRun AlCaRecoTriggerBitsRcdRead_TEMPL_cfg.py inputDB=sqlite_file:AlCaRecoHLTpaths_TEST.db inputTag=AlCaRecoHLTpaths_TEST
```
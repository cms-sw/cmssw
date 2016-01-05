import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_1.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_2.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_3.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_4.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_5.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_6.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_7.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_8.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_9.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_10.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_11.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_12.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_13.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_14.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_15.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_16.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_17.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_18.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_19.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_20.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_21.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_22.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_23.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_24.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_25.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_26.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_27.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_28.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_29.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_30.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_31.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_32.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_33.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_34.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_35.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_36.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_37.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_38.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_39.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_40.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_41.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_42.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_43.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_44.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_45.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_46.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_47.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_48.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_49.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_50.root',
       'file:///afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1688/iso_mc_sample/tree_51.root' 
       ]);


secFiles.extend( [
               ] )



# preliminary b-tagging Operating Points
# obtained with cmssw_2_1_0_pre6
# qcd validation /store/relval/2008/6/22/RelVal-RelValQCD_Pt_80_120-1213987236-IDEAL_V2-2nd/0003/
# corrected pt 30 |eta| <2.4 taggability >2
#

import FWCore.ParameterSet.Config as cms

BJetOperatingPointsParameters = cms.PSet(
   BJetOperatingPoints = cms.PSet(
      DefaultBdisc = cms.string('trackCountingHighEffBJetTags'),
      DefaultOp = cms.string('Loose'),
        discCutTight = cms.vdouble(
            13.76,  3.943,         #TCHE, TCHP,
            0.7322, 3.335,         #JTP,  JBTP, 
            3.524,  0.9467,        #SSV,  CSV, 
            0.9635, 0.9462,        #MSV,  IPM,
            0.5581, 0.2757, 0.349  #SET,  SMT,  SMNoIPT  
        ),
        discCutMedium = cms.vdouble(
            4.433,  2.53,
            0.5114, 2.295,
            2.13,   0.8339,
            0.8131, 0.8141,
            0.1974, 0.1208, 0.1846
        ),
        discCutLoose = cms.vdouble(
            1.993,  1.678,
            0.2395, 1.149,
            1.2,    0.415,
            0.4291, 0.3401,
            0.0,    0.0,    0.0
        ),
        bdiscriminators = cms.vstring(
            'trackCountingHighEffBJetTags','trackCountingHighPurBJetTags',
            'jetProbabilityBJetTags','jetBProbabilityBJetTags',
            'simpleSecondaryVertexBJetTags','combinedSecondaryVertexBJetTags',
            'combinedSecondaryVertexMVABJetTags','impactParameterMVABJetTags',
            'softElectronBJetTags','softMuonBJetTags','softMuonNoIPBJetTags'
        )
   )
)

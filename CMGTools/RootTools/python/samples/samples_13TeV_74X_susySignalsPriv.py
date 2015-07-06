#####COMPONENT CREATOR
from CMGTools.RootTools.samples.ComponentCreator import ComponentCreator
kreator = ComponentCreator()

T5ttttDeg_mGo1000_mStop300_mCh285_mChi280 = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1000_mStop300_mCh285_mChi280', '/T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_23bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388)
#T5ttttDeg_mGo1300_mStop300_mCh285_mChi280 = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1300_mStop300_mCh285_mChi280', '/T5ttttDeg_mGo1300_mStop300_mCh285_mChi280_23bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0460525)
T5ttttDeg_mGo1000_mStop300_mChi280 = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1000_mStop300_mChi280', '/T5ttttDeg_mGo1000_mStop300_mChi280_4bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388)
#T5ttttDeg_mGo1300_mStop300_mChi280 = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1300_mStop300_mChi280', '/T5ttttDeg_mGo1300_mStop300_mChi280_4bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0460525)
#T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_dil = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_dil', '/T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_23bodydec_dilepfilterPt8p5_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388)
#T5ttttDeg_mGo1300_mStop300_mCh285_mChi280_dil = kreator.makeMCComponentFromEOS('T5ttttDeg_mGo1300_mStop300_mCh285_mChi280_dil', '/T5ttttDeg_mGo1300_mStop300_mCh285_mChi280_23bodydec_dilepfilterPt8p5_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0460525)
T5ttttDeg = [ T5ttttDeg_mGo1000_mStop300_mCh285_mChi280, T5ttttDeg_mGo1000_mStop300_mChi280 ]

#T1ttbbWW_mGo1000_mCh725_mChi715 = kreator.makeMCComponentFromEOS('T1ttbbWW_mGo1000_mCh725_mChi715', '/T1ttbbWW_2J_mGo1000_mCh725_mChi715_3bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388)
#T1ttbbWW_mGo1000_mCh725_mChi720 = kreator.makeMCComponentFromEOS('T1ttbbWW_mGo1000_mCh725_mChi720', '/T1ttbbWW_2J_mGo1000_mCh725_mChi720_3bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388)
#T1ttbbWW_mGo1300_mCh300_mChi290 = kreator.makeMCComponentFromEOS('T1ttbbWW_mGo1300_mCh300_mChi290', '/T1ttbbWW_2J_mGo1300_mCh300_mChi290_3bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0460525)
#T1ttbbWW_mGo1300_mCh300_mChi295 = kreator.makeMCComponentFromEOS('T1ttbbWW_mGo1300_mCh300_mChi295', '/T1ttbbWW_2J_mGo1300_mCh300_mChi295_3bodydec_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0460525)
T1ttbbWW = [ ] # T1ttbbWW_mGo1000_mCh725_mChi715, T1ttbbWW_mGo1000_mCh725_mChi720, T1ttbbWW_mGo1300_mCh300_mChi290, T1ttbbWW_mGo1300_mCh300_mChi295 ]

#T1ttbb_mGo1500_mChi100 = kreator.makeMCComponentFromEOS('T1ttbb_mGo1500_mChi100', '/T1ttbb_2J_mGo1500_mChi100_3bodydec_asymmDecOnly/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0141903)
T1ttbb = [ ] #T1ttbb_mGo1500_mChi100 ]

T6ttWW_mSbot600_mCh425_mChi50 = kreator.makeMCComponentFromEOS('T6ttWW_mSbot600_mCh425_mChi50', '/T6ttWW_600_425_50_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.174599)
T6ttWW_mSbot650_mCh150_mChi50 = kreator.makeMCComponentFromEOS('T6ttWW_mSbot650_mCh150_mChi50', '/T6ttWW_650_150_50_v2/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.107045)
T6ttWW = [ T6ttWW_mSbot600_mCh425_mChi50, T6ttWW_mSbot650_mCh150_mChi50 ]

#SqGltttt_mGo1300_mSq1300_mChi100 = kreator.makeMCComponentFromEOS('SqGltttt_mGo1300_mSq1300_mChi100', '/13TeV_SqGltttt_Gl_1300_Sq_1300_LSP_100/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root')
SqGltttt = [ ] #SqGltttt_mGo1300_mSq1300_mChi100 ]

T1tttt_mGo1500_mChi100 = kreator.makeMCComponentFromEOS('T1tttt_mGo1500_mChi100', '/T1tttt_mGo1500_mChi100/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s', '.*root', 0.0141903)
T1tttt_mGo1200_mChi800 = kreator.makeMCComponentFromEOS('T1tttt_mGo1200_mChi800', '/T1tttt_mGo1200_mChi800/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s', '.*root', 0.0856418)
T1tttt_priv = [ T1tttt_mGo1500_mChi100, T1tttt_mGo1200_mChi800 ] 

#T5qqqqWWDeg_mGo1400_mCh315_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1400_mCh315_mChi300', '/SMS_T5qqqqWW_mGl1400_mChi315_mLSP300/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 0.0252977)
#T5qqqqWWDeg_mGo1000_mCh310_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh310_mChi300', '/T5qqqqWWDeg_mGo1000_mCh310_mChi300/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388) 
#T5qqqqWWDeg_mGo1000_mCh310_mChi300_dilep= kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh310_mChi300_dilep', '/T5qqqqWWDeg_mGo1000_mCh310_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388*(0.333)*(0.333)) 
#T5qqqqWWDeg_mGo1000_mCh315_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh315_mChi300', '/T5qqqqWWDeg_mGo1000_mCh315_mChi300/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388) 
T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep', '/T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388*(0.333)*(0.333)) 
#T5qqqqWWDeg_mGo1000_mCh325_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh325_mChi300', '/T5qqqqWWDeg_mGo1000_mCh325_mChi300/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388) 
T5qqqqWWDeg_mGo1000_mCh325_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo1000_mCh325_mChi300_dilep', '/T5qqqqWWDeg_mGo1000_mCh325_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388*(0.324)*(0.324)) 
#T5qqqqWWDeg_mGo800_mCh305_mChi300 = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo800_mCh305_mChi300', '/T5qqqqWWDeg_mGo800_mCh305_mChi300/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 1.4891) 
#T5qqqqWWDeg_mGo800_mCh305_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqWWDeg_mGo800_mCh305_mChi300_dilep', '/T5qqqqWWDeg_mGo800_mCh305_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 1.4891*(0.342)*(0.342)) 
T5qqqqWWDeg = [
    #T5qqqqWWDeg_mGo1400_mCh315_mChi300,
    #5qqqqWWDeg_mGo1000_mCh310_mChi300, T5qqqqWWDeg_mGo1000_mCh315_mChi300, T5qqqqWWDeg_mGo1000_mCh325_mChi300, T5qqqqWWDeg_mGo800_mCh305_mChi300,
    #T5qqqqWWDeg_mGo1000_mCh310_mChi300_dilep, T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep, T5qqqqWWDeg_mGo1000_mCh325_mChi300_dilep, T5qqqqWWDeg_mGo800_mCh305_mChi300_dilep
    T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep, T5qqqqWWDeg_mGo1000_mCh325_mChi300_dilep
]

T5qqqqWZDeg_mGo1000_mCh315_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqWZDeg_mGo1000_mCh315_mChi300_dilep', '/T5qqqqWZDeg_mGo1000_mCh315_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388*(0.333)*(0.112)) 
T5qqqqWZDeg_mGo1000_mCh325_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqWZDeg_mGo1000_mCh325_mChi300_dilep', '/T5qqqqWZDeg_mGo1000_mCh325_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388*(0.324)*(0.108)) 
T5qqqqWZDeg = [
    T5qqqqWZDeg_mGo1000_mCh315_mChi300_dilep, T5qqqqWZDeg_mGo1000_mCh325_mChi300_dilep
]
T5qqqqZZDeg_mGo1000_mCh315_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqZZDeg_mGo1000_mCh315_mChi300_dilep', '/T5qqqqZZDeg_mGo1000_mCh315_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388*(0.112)*(0.112)) 
T5qqqqZZDeg_mGo1000_mCh325_mChi300_dilep = kreator.makeMCComponentFromEOS('T5qqqqZZDeg_mGo1000_mCh325_mChi300_dilep', '/T5qqqqZZDeg_mGo1000_mCh325_mChi300_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388*(0.108)*(0.108)) 
T5qqqqZZDeg = [
    T5qqqqZZDeg_mGo1000_mCh315_mChi300_dilep, T5qqqqZZDeg_mGo1000_mCh325_mChi300_dilep
]

T5qqqqVVDeg = T5qqqqWWDeg + T5qqqqWZDeg + T5qqqqZZDeg


#T5qqqqWW_mGo1500_mCh800_mChi100 = kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1500_mCh800_mChi100', '/SMS_T5qqqqWW_Gl1500_Chi800_LSP100/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 0.0141903)
#T5qqqqWW_mGo1200_mCh1000_mChi800 = kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1200_mCh1000_mChi800', '/SMS_T5qqqqWW_Gl1200_Chi1000_LSP800/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 0.0856418)
#T5qqqqWW_mGo1000_mCh800_mChi700 = kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1000_mCh800_mChi700', '/T5qqqqWW_mGo1000_mCh800_mChi700/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388) 
#T5qqqqWW_mGo1000_mCh800_mChi700_dilep= kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1000_mCh800_mChi700_dilep', '/T5qqqqWW_mGo1000_mCh800_mChi700_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.325388*(3*0.108)*(3*0.108)) 
#T5qqqqWW_mGo1200_mCh1000_mChi800_cmg = kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1200_mCh1000_mChi800_cmg', '/T5qqqqWW_mGo1200_mCh1000_mChi800/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0856418) 
T5qqqqWW_mGo1200_mCh1000_mChi800_dilep= kreator.makeMCComponentFromEOS('T5qqqqWW_mGo1200_mCh1000_mChi800_dilep', '/T5qqqqWW_mGo1200_mCh1000_mChi800_dilep/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0856418*(3*0.108)*(3*0.108)) 

T5qqqqWW = [
    #T5qqqqWW_mGo1500_mCh800_mChi100, T5qqqqWW_mGo1200_mCh1000_mChi800,
    #T5qqqqWW_mGo1000_mCh800_mChi700, T5qqqqWW_mGo1200_mCh1000_mChi800_cmg,
    #T5qqqqWW_mGo1000_mCh800_mChi700_dilep, 
    T5qqqqWW_mGo1200_mCh1000_mChi800_dilep
]

T5qqqqWZ_mGo1200_mCh1000_mChi800_dilep = kreator.makeMCComponentFromEOS('T5qqqqWZ_mGo1200_mCh1000_mChi800_dilep', '/T5qqqqWZ_mGo1200_mCh1000_mChi800/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0141903*(3*0.108)*(3*0.03366)) 
T5qqqqWZ_mGo1500_mCh800_mChi100_dilep = kreator.makeMCComponentFromEOS('T5qqqqWZ_mGo1500_mCh800_mChi100_dilep', '/T5qqqqWZ_mGo1500_mCh800_mChi100/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0856418*(3*0.108)*(3*0.03366)) 
T5qqqqWZ = [
    T5qqqqWZ_mGo1500_mCh800_mChi100_dilep, T5qqqqWZ_mGo1200_mCh1000_mChi800_dilep
]

T5qqqqZZ_mGo1200_mCh1000_mChi800_dilep = kreator.makeMCComponentFromEOS('T5qqqqZZ_mGo1200_mCh1000_mChi800_dilep', '/T5qqqqZZ_mGo1200_mCh1000_mChi800/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0141903*(3*0.03366)*(3*0.03366)) 
T5qqqqZZ_mGo1500_mCh800_mChi100_dilep = kreator.makeMCComponentFromEOS('T5qqqqZZ_mGo1500_mCh800_mChi100_dilep', '/T5qqqqZZ_mGo1500_mCh800_mChi100/', '/store/cmst3/group/susy/gpetrucc/13TeV/RunIISpring15DR74/%s',".*root", 0.0856418*(3*0.03366)*(3*0.03366)) 
T5qqqqZZ = [
    T5qqqqZZ_mGo1500_mCh800_mChi100_dilep, T5qqqqZZ_mGo1200_mCh1000_mChi800_dilep
]


T5qqqqVV = T5qqqqWW + T5qqqqWZ + T5qqqqZZ


# note: cross section for q~ q~ from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SUSYCrossSections13TeVsquarkantisquark (i.e. gluinos and stops decoupled)
#T6qqWW_mSq950_mCh325_mChi300 = kreator.makeMCComponentFromEOS('T6qqWW_mSq950_mCh325_mChi300', '/SMS_T6qqWW_mSq950_mChi325_mLSP300/', '/store/cmst3/group/susy/alobanov/MC/PHYS14/PU20_25ns/%s', '.*root', 0.0898112)
T6qqWW = [ ] # T6qqWW_mSq950_mCh325_mChi300 ]


mcSamplesPriv = T5ttttDeg + T1ttbbWW + T1ttbb + T6ttWW + SqGltttt + T1tttt_priv + T5qqqqVV + T5qqqqVVDeg + T6qqWW


from CMGTools.TTHAnalysis.setup.Efficiencies import *
dataDir = "$CMSSW_BASE/src/CMGTools/TTHAnalysis/data"

#Define splitting
for comp in mcSamplesPriv:
    comp.isMC = True
    comp.isData = False
    comp.splitFactor = 250 #  if comp.name in [ "WJets", "DY3JetsM50", "DY4JetsM50","W1Jets","W2Jets","W3Jets","W4Jets","TTJetsHad" ] else 100
    comp.puFileMC=dataDir+"/puProfile_Summer12_53X.root"
    comp.puFileData=dataDir+"/puProfile_Data12.root"
    comp.efficiency = eff2012


if __name__ == "__main__":
   import sys
   if "test" in sys.argv:
       from CMGTools.RootTools.samples.ComponentCreator import testSamples
       testSamples(mcSamplesPriv)

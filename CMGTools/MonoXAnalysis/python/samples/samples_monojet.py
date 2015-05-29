import PhysicsTools.HeppyCore.framework.config as cfg
import os
dataDir = os.environ['CMSSW_BASE']+"/src/CMGTools/TTHAnalysis/data"

################## Triggers   
from CMGTools.MonoXAnalysis.samples.triggers_monojet import *



#####COMPONENT CREATOR#####

from CMGTools.TTHAnalysis.samples.ComponentCreator import ComponentCreator
kreator = ComponentCreator()

################################################################################
#### Background samples
################## PU20 bx25ns (default of phys14, so no postfix) ##############

# W inclusive (cross section from FEWZ, StandardModelCrossSectionsat13TeV)
WJetsToLNu = kreator.makeMCComponent("WJetsToLNu","/WJetsToLNu_13TeV-madgraph-pythia8-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 20508.9)

#W+jets 
# cross sections for WJets taken from McM LO times inclusive k-factor from FEWZ(20508.9 pb x3)/McM(50100.0)
WJetsToLNu_HT100to200 = kreator.makeMCComponent("WJetsToLNu_HT100to200", "/WJetsToLNu_HT-100to200_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",1817.0*1.23)
WJetsToLNu_HT200to400 = kreator.makeMCComponent("WJetsToLNu_HT200to400", "/WJetsToLNu_HT-200to400_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",471.6*1.23)
WJetsToLNu_HT400to600 = kreator.makeMCComponent("WJetsToLNu_HT400to600", "/WJetsToLNu_HT-400to600_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",55.61*1.23)
WJetsToLNu_HT600toInf = kreator.makeMCComponent("WJetsToLNu_HT600toInf", "/WJetsToLNu_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",18.81*1.23)
WJetsToLNuHT = [
WJetsToLNu_HT100to200,
WJetsToLNu_HT200to400,
WJetsToLNu_HT400to600,
WJetsToLNu_HT600toInf,
]


# DY inclusive (cross section from FEWZ, StandardModelCrossSectionsat13TeV)
DYJetsToLL_M50 = kreator.makeMCComponent("DYJetsToLL_M50", "/DYJetsToLL_M-50_13TeV-madgraph-pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 2008.*3)

# DY HT bins: cross sections for DYJets taken from McM LO times inclusive k-factor from FEWZ(2008 pb x3)/McM(4746)
DYJetsToLL_M50_HT100to200 = kreator.makeMCComponent("DYJetsToLL_M50_HT100to200", "/DYJetsToLL_M-50_HT-100to200_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",194.3*1.27)
DYJetsToLL_M50_HT200to400 = kreator.makeMCComponent("DYJetsToLL_M50_HT200to400", "/DYJetsToLL_M-50_HT-200to400_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",52.24*1.27)
DYJetsToLL_M50_HT400to600 = kreator.makeMCComponent("DYJetsToLL_M50_HT400to600", "/DYJetsToLL_M-50_HT-400to600_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",6.546*1.27)
DYJetsToLL_M50_HT600toInf = kreator.makeMCComponent("DYJetsToLL_M50_HT600toInf", "/DYJetsToLL_M-50_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",2.179*1.27)
DYJetsM50HT = [
DYJetsToLL_M50_HT100to200,
DYJetsToLL_M50_HT200to400,
DYJetsToLL_M50_HT400to600,
DYJetsToLL_M50_HT600toInf,
]

#Z-> nunu +jets
ZJetsToNuNu_HT100to200 = kreator.makeMCComponent("ZJetsToNuNu_HT100to200", "/ZJetsToNuNu_HT-100to200_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",372.6*1.27)
ZJetsToNuNu_HT200to400 = kreator.makeMCComponent("ZJetsToNuNu_HT200to400", "/ZJetsToNuNu_HT-200to400_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",100.8*1.27)
ZJetsToNuNu_HT400to600 = kreator.makeMCComponent("ZJetsToNuNu_HT400to600", "/ZJetsToNuNu_HT-400to600_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root",11.99*1.27)
ZJetsToNuNu_HT600toInf = kreator.makeMCComponent("ZJetsToNuNu_HT600toInf", "/ZJetsToNuNu_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",4.113*1.27)
ZJetsToNuNuHT = [
ZJetsToNuNu_HT100to200,
ZJetsToNuNu_HT200to400,
ZJetsToNuNu_HT400to600,
ZJetsToNuNu_HT600toInf,
]


#photon+jets
GJets_HT100to200 = kreator.makeMCComponent("GJets_HT100to200", "/GJets_HT-100to200_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",1534)
GJets_HT200to400 = kreator.makeMCComponent("GJets_HT200to400", "/GJets_HT-200to400_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",489.9)
GJets_HT400to600 = kreator.makeMCComponent("GJets_HT400to600", "/GJets_HT-400to600_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",62.05)
GJets_HT600toInf = kreator.makeMCComponent("GJets_HT600toInf", "/GJets_HT-600toInf_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",20.87)
GJetsHT = [
GJets_HT100to200,
GJets_HT200to400,
GJets_HT400to600,
GJets_HT600toInf,
]

#photon+jets relval for synch, dummy xsec
GJets_RelVal_PT10 = kreator.makeMCComponent("GJets_RelVal", "/RelValPhotonJets_Pt_10_13/CMSSW_7_0_6_patch1-PLS170_V6AN1-v1/MINIAODSIM", "CMS", ".*root", 1)
GJets_RelVal = [GJets_RelVal_PT10]

#QCD 
## Cross sections from McM (LO Madgraph)
QCD_HT_100To250 = kreator.makeMCComponent("QCD_HT_100To250", "/QCD_HT-100To250_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 28730000)
QCD_HT_250To500 = kreator.makeMCComponent("QCD_HT_250To500", "/QCD_HT_250To500_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 670500)
QCD_HT_250To500_ext1 = kreator.makeMCComponent("QCD_HT_250To500_ext1", "/QCD_HT_250To500_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1_ext1-v2/MINIAODSIM", "CMS", ".*root", 670500)
QCD_HT_500To1000 = kreator.makeMCComponent("QCD_HT_500To1000", "/QCD_HT-500To1000_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 26740)
QCD_HT_500To1000_ext1 = kreator.makeMCComponent("QCD_HT_500To1000_ext1", "/QCD_HT-500To1000_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1_ext1-v1/MINIAODSIM", "CMS", ".*root", 26740)
QCD_HT_1000ToInf_ext1 = kreator.makeMCComponent("QCD_HT_1000ToInf_ext1", "/QCD_HT_1000ToInf_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1_ext1-v1/MINIAODSIM", "CMS", ".*root", 769.7)
QCD_HT_1000ToInf = kreator.makeMCComponent("QCD_HT_1000ToInf", "/QCD_HT_1000ToInf_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 769.7)
QCDHT = [
QCD_HT_100To250,
QCD_HT_250To500,
QCD_HT_500To1000,
QCD_HT_1000ToInf,
QCD_HT_250To500_ext1,
QCD_HT_500To1000_ext1,
QCD_HT_1000ToInf_ext1
]

QCD_Pt15to30 = kreator.makeMCComponent("QCD_Pt15to30","/QCD_Pt-15to30_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 2237000000)
QCD_Pt30to50 = kreator.makeMCComponent("QCD_Pt30to50","/QCD_Pt-30to50_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 161500000)
QCD_Pt50to80 = kreator.makeMCComponent("QCD_Pt50to80","/QCD_Pt-50to80_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 22110000)
QCD_Pt80to120 = kreator.makeMCComponent("QCD_Pt80to120","/QCD_Pt-80to120_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 3000114.3)
QCD_Pt120to170 = kreator.makeMCComponent("QCD_Pt120to170","/QCD_Pt-120to170_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 493200)
QCD_Pt170to300 = kreator.makeMCComponent("QCD_Pt170to300","/QCD_Pt-170to300_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 120300)
QCD_Pt300to470 = kreator.makeMCComponent("QCD_Pt300to470","/QCD_Pt-300to470_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 7475)
QCD_Pt470to600 = kreator.makeMCComponent("QCD_Pt470to600","/QCD_Pt-470to600_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 587.1)
QCD_Pt600to800 = kreator.makeMCComponent("QCD_Pt600to800","/QCD_Pt-600to800_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 167)
QCD_Pt800to1000 = kreator.makeMCComponent("QCD_Pt800to1000","/QCD_Pt-800to1000_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 28.25)
QCD_Pt1000to1400 = kreator.makeMCComponent("QCD_Pt1000to1400","/QCD_Pt-1000to1400_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 8.195)
QCD_Pt1400to1800 = kreator.makeMCComponent("QCD_Pt1400to1800","/QCD_Pt-1400to1800_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 0.7346)
QCD_Pt1800to2400 = kreator.makeMCComponent("QCD_Pt1800to2400","/QCD_Pt-1800to2400_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 0.102, True)
QCD_Pt2400to3200 = kreator.makeMCComponent("QCD_Pt2400to3200","/QCD_Pt-2400to3200_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 0.00644, True)
QCD_Pt3200 = kreator.makeMCComponent("QCD_Pt3200","/QCD_Pt-3200_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_trkalmb_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 0.000163, True)

QCDPt = [
QCD_Pt15to30,
QCD_Pt30to50,
QCD_Pt50to80,
QCD_Pt80to120,
QCD_Pt120to170,
QCD_Pt170to300,
QCD_Pt300to470,
QCD_Pt470to600,
QCD_Pt600to800,
QCD_Pt800to1000,
QCD_Pt1000to1400,
QCD_Pt1400to1800,
QCD_Pt1800to2400,
QCD_Pt2400to3200,
QCD_Pt3200
]

# Muon-enriched QCD (cross sections and filter efficiency from McM)
QCD_Mu15 = kreator.makeMCComponent("QCD_Mu15", "/QCD_Pt-20toInf_MuEnrichedPt15_PionKaonDecay_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v3/MINIAODSIM", "CMS", ".*root", 866.6e6*0.00044);

# Electron-enriched QCD (cross sections and filter efficiency from McM)
QCD_Pt10to20_EMEnriched = kreator.makeMCComponent("QCD_Pt10to20_EMEnriched", "/QCD_Pt-10to20_EMEnriched_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 8838e6*0.143);
QCD_Pt20to30_EMEnriched = kreator.makeMCComponent("QCD_Pt20to30_EMEnriched", "/QCD_Pt-20to30_EMEnriched_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_castor_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 6773e5*0.007);
QCD_Pt30to80_EMEnriched = kreator.makeMCComponent("QCD_Pt30to80_EMEnriched", "/QCD_Pt-30to80_EMEnriched_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 1859e5*0.056);
QCD_Pt80to170_EMEnriched = kreator.makeMCComponent("QCD_Pt80to170_EMEnriched", "/QCD_Pt-80to170_EMEnriched_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_castor_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 3259e3*0.158);
QCDPtEMEnriched = [
QCD_Pt10to20_EMEnriched,
QCD_Pt20to30_EMEnriched,
QCD_Pt30to80_EMEnriched,
QCD_Pt80to170_EMEnriched
]

QCD_Pt20to30_bcToE = kreator.makeMCComponent("QCD_Pt20to30_bcToE", "/QCD_Pt_20to30_bcToE_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 6759e5*0.00075);
QCD_Pt30to80_bcToE = kreator.makeMCComponent("QCD_Pt30to80_bcToE", "/QCD_Pt_30to80_bcToE_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 1859e5*0.00272);
QCD_Pt80to170_bcToE = kreator.makeMCComponent("QCD_Pt80to170_bcToE", "/QCD_Pt_80to170_bcToE_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root", 3495e3*0.01225);
QCD_Pt170toInf_bcToE = kreator.makeMCComponent("QCD_Pt170toInf_bcToE", "/QCD_Pt_170toInf_bcToE_Tune4C_13TeV_pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 1285e2*0.0406);
QCDPtbcToE = [
QCD_Pt20to30_bcToE,
QCD_Pt30to80_bcToE,
QCD_Pt80to170_bcToE,
QCD_Pt170toInf_bcToE
]



# Single top cross sections: https://twiki.cern.ch/twiki/bin/viewauth/CMS/SingleTopSigma
TToLeptons_tch = kreator.makeMCComponent("TToLeptons_tch", "/TToLeptons_t-channel-CSA14_Tune4C_13TeV-aMCatNLO-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 136.05*0.108*3)
TToLeptons_sch = kreator.makeMCComponent("TToLeptons_sch", "/TToLeptons_s-channel-CSA14_Tune4C_13TeV-aMCatNLO-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 7.20*0.108*3)
TBarToLeptons_tch = kreator.makeMCComponent("TBarToLeptons_tch", "/TBarToLeptons_t-channel_Tune4C_CSA14_13TeV-aMCatNLO-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 80.97*0.108*3)
TBarToLeptons_sch = kreator.makeMCComponent("TBarToLeptons_sch", "/TBarToLeptons_s-channel-CSA14_Tune4C_13TeV-aMCatNLO-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",  4.16*0.108*3)
TBar_tWch = kreator.makeMCComponent("TBar_tWch", "/Tbar_tW-channel-DR_Tune4C_13TeV-CSA14-powheg-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",35.6)
T_tWch = kreator.makeMCComponent("T_tWch", "/T_tW-channel-DR_Tune4C_13TeV-CSA14-powheg-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",35.6)

SingleTop = [
    TToLeptons_tch, TToLeptons_sch, TBarToLeptons_tch, TBarToLeptons_sch, TBar_tWch, T_tWch
]

# TTbar cross section: MCFM with dynamic scale, StandardModelCrossSectionsat13TeV
TTJets = kreator.makeMCComponent("TTJets", "/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",809.1)

# TTV cross sections are from 13 TeV MG5_aMC@NLO v2.2.1, NNPDF 2.3nlo, fixed scale = mtop + 0.5*mv
TTWJets = kreator.makeMCComponent("TTWJets", "/TTWJets_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.6647)
TTZJets = kreator.makeMCComponent("TTZJets", "/TTZJets_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",0.8565)

# TTH cross section from LHC Higgs XS WG: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt1314TeV?rev=15
TTH = kreator.makeMCComponent("TTH", "/TTbarH_M-125_13TeV_amcatnlo-pythia8-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v2/MINIAODSIM", "CMS", ".*root",0.5085)


# cross section from StandardModelCrossSectionsat13TeV (NLO MCFM, mll > 12) times BR=(3*0.108)*(3*0.0337)
WZJetsTo3LNu = kreator.makeMCComponent("WZJetsTo3LNu", "/WZJetsTo3LNu_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",2.165)
# cross section from StandardModelCrossSectionsat13TeV (NLO MCFM, mll > 12) times BR=(3*0.0337)**2
ZZTo4L = kreator.makeMCComponent("ZZTo4L","/ZZTo4L_Tune4C_13TeV-powheg-pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",  31.8*(3*0.03366**2))

# GGH cross section from LHC Higgs XS WG: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt1314TeV?rev=15
GGHZZ4L = kreator.makeMCComponent("GGHZZ4L", "/GluGluToHToZZTo4L_M-125_13TeV-powheg-pythia6/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root", 43.92*2.76E-04)



#### Signal samples
# cross sections from https://twiki.cern.ch/twiki/bin/viewauth/CMS/Monojet
Monojet_M_10_V = kreator.makeMCComponent("Monojet_M_10_V", '/DarkMatter_Monojet_M-10_V_Tune4C_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM', "CMS", ".*root", 9.5463E-07)
Monojet_M_100_V = kreator.makeMCComponent("Monojet_M_100_V", '/DarkMatter_Monojet_M-100_V_Tune4C_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM', "CMS", ".*root", 9.0451E-07)
Monojet_M_1000_V = kreator.makeMCComponent("Monojet_M_1000_V", '/DarkMatter_Monojet_M-1000_V_Tune4C_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM', "CMS", ".*root", 1.2438E-07)
Monojet_M_1_AV = kreator.makeMCComponent("Monojet_M_1_AV", '/DarkMatter_Monojet_M-1_AV_Tune4C_13TeV-madgraph/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM', "CMS", ".*root", 9.5695E-07)
Monojet_M_10_AV = kreator.makeMCComponent("Monojet_M_10_AV", '/DarkMatter_Monojet_M-10_AV_Tune4C_13TeV-madgraph/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM', "CMS", ".*root", 9.5381E-07)
Monojet_M_100_AV = kreator.makeMCComponent("Monojet_M_100_AV", '/DarkMatter_Monojet_M-100_AV_Tune4C_13TeV-madgraph/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM', "CMS", ".*root", 8.0087E-07)
Monojet_M_1000_AV = kreator.makeMCComponent("Monojet_M_1000_AV", '/DarkMatter_Monojet_M-1000_AV_Tune4C_13TeV-madgraph/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM', "CMS", ".*root", 4.6629E-07)
MonojetSignalSamples = [
Monojet_M_10_V,
Monojet_M_100_V,
Monojet_M_1000_V,
Monojet_M_1_AV,
Monojet_M_10_AV,
Monojet_M_100_AV,
Monojet_M_1000_AV,
]

mcSamplesPHYS14_PU20bx25 = QCDHT + QCDPt + [QCD_Mu15] + QCDPtEMEnriched + QCDPtbcToE + [WJetsToLNu] + WJetsToLNuHT +  [DYJetsToLL_M50] + DYJetsM50HT + GJetsHT + ZJetsToNuNuHT + SingleTop + [ TTJets, TTWJets, TTZJets, WZJetsTo3LNu] + MonojetSignalSamples

## PRIVATE SAMPLES

GJets_HT100to200_fixPhoton = kreator.makeMCComponentFromEOS('GJets_HT100to200', '/GJets_HT-100to200_Tune4C_13TeV-madgraph-tauola/miniAOD_fixPhoton_7_2_3/150204_164703/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s',".*root", 1534)
GJets_HT200to400_fixPhoton = kreator.makeMCComponentFromEOS('GJets_HT200to400', '/GJets_HT-200to400_Tune4C_13TeV-madgraph-tauola/miniAOD_fixPhoton_7_2_3/150204_164621/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s',".*root", 489.9)
GJets_HT400to600_fixPhoton = kreator.makeMCComponentFromEOS('GJets_HT400to600', '/GJets_HT-400to600_Tune4C_13TeV-madgraph-tauola/miniAOD_fixPhoton_7_2_3/150204_164547/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s',".*root", 62.05)
GJets_HT600toInf_fixPhoton = kreator.makeMCComponentFromEOS('GJets_HT600toInf', '/GJets_HT-600toInf_Tune4C_13TeV-madgraph-tauola/miniAOD_fixPhoton_7_2_3/150204_122016/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s',".*root", 20.87)
GJets_fixPhoton = [
GJets_HT100to200_fixPhoton,
GJets_HT200to400_fixPhoton,
GJets_HT400to600_fixPhoton,
GJets_HT600toInf_fixPhoton,
]

QCD_HT_100To250_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_100To250", '/QCD_HT-100To250_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_145121/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s',  ".*root", 28730000)
QCD_HT_250To500_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_250To500", '/QCD_HT_250To500_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_145040/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s',  ".*root", 670500)
QCD_HT_250To500_ext1_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_250To500_ext1", '/QCD_HT_250To500_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_144831/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s', ".*root", 670500)
QCD_HT_500To1000_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_500To1000", '/QCD_HT-500To1000_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_144759/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s', ".*root", 26740)
QCD_HT_500To1000_ext1_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_500To1000_ext1", '/QCD_HT-500To1000_13TeV-madgraph/miniAOD_fixPhoton_reco/150206_144615/0000/', '/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_reco/%s', ".*root", 26740)
QCD_HT_1000ToInf_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_1000ToInf", '/QCD_HT_1000ToInf_13TeV-madgraph/miniAOD_fixPhoton_7_2_3/150204_172505/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s', ".*root", 769.7)
QCD_HT_1000ToInf_ext1_fixPhoton = kreator.makeMCComponentFromEOS("QCD_HT_1000ToInf_ext1", '/QCD_HT_1000ToInf_13TeV-madgraph/miniAOD_fixPhoton_7_2_3/150204_172427/0000/', '/store/user/mmasciov/PHYS14_fixPhoton/%s', ".*root", 769.7)
QCDHT_fixPhoton = [
QCD_HT_100To250_fixPhoton,
QCD_HT_250To500_fixPhoton,
QCD_HT_500To1000_fixPhoton,
QCD_HT_1000ToInf_fixPhoton,
QCD_HT_250To500_ext1_fixPhoton,
QCD_HT_500To1000_ext1_fixPhoton,
QCD_HT_1000ToInf_ext1_fixPhoton,
]

QCD_Pt170to300_fixPhoton   = kreator.makeMCComponentFromEOS("QCD_Pt170to300"  , "QCD_Pt-170to300_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154438/0000/"  , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 120300)
QCD_Pt300to470_fixPhoton   = kreator.makeMCComponentFromEOS("QCD_Pt300to470"  , "QCD_Pt-300to470_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154529/0000/"  , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 7475)
QCD_Pt470to600_fixPhoton   = kreator.makeMCComponentFromEOS("QCD_Pt470to600"  , "QCD_Pt-470to600_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154700/0000/"  , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 587.1)
QCD_Pt600to800_fixPhoton   = kreator.makeMCComponentFromEOS("QCD_Pt600to800"  , "QCD_Pt-600to800_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154904/0000/"  , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 167)
QCD_Pt800to1000_fixPhoton  = kreator.makeMCComponentFromEOS("QCD_Pt800to1000" , "QCD_Pt-800to1000_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_155003/0000/" , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 28.25)
QCD_Pt1000to1400_fixPhoton = kreator.makeMCComponentFromEOS("QCD_Pt1000to1400", "QCD_Pt-1000to1400_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154248/0000/", "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 8.195)
QCD_Pt1400to1800_fixPhoton = kreator.makeMCComponentFromEOS("QCD_Pt1400to1800", "QCD_Pt-1400to1800_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150228_154344/0000/", "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 0.7346)
QCD_Pt1800to2400_fixPhoton = kreator.makeMCComponentFromEOS("QCD_Pt1800to2400", "QCD_Pt-1800to2400_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150301_002302/0000/", "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 0.102)
QCD_Pt2400to3200_fixPhoton = kreator.makeMCComponentFromEOS("QCD_Pt2400to3200", "QCD_Pt-2400to3200_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150301_002547/0000/", "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 0.00644)
QCD_Pt3200_fixPhoton       = kreator.makeMCComponentFromEOS("QCD_Pt3200"      , "QCD_Pt-3200_Tune4C_13TeV_pythia8/miniAOD_fixPhoton_QCDPt/150301_002653/0000/"      , "/store/group/phys_susy/mmasciov/PHYS14_fixPhoton_QCDPt/%s", ".*root", 0.000163)


QCDPt_fixPhoton = [
QCD_Pt170to300_fixPhoton,
QCD_Pt300to470_fixPhoton,
QCD_Pt470to600_fixPhoton,
QCD_Pt600to800_fixPhoton,
QCD_Pt800to1000_fixPhoton,
QCD_Pt1000to1400_fixPhoton,
QCD_Pt1400to1800_fixPhoton,
QCD_Pt1800to2400_fixPhoton,
QCD_Pt2400to3200_fixPhoton,
QCD_Pt3200_fixPhoton
]


mcSamplesPriv = GJets_fixPhoton + QCDHT_fixPhoton

mcSamples = mcSamplesPHYS14_PU20bx25 + mcSamplesPriv

from CMGTools.TTHAnalysis.setup.Efficiencies import *


#Define splitting
for comp in mcSamples:
    comp.isMC = True
    comp.isData = False
    comp.splitFactor = 250 #  if comp.name in [ "WJets", "DY3JetsM50", "DY4JetsM50","W1Jets","W2Jets","W3Jets","W4Jets","TTJetsHad" ] else 100
    comp.puFileMC=dataDir+"/puProfile_Summer12_53X.root"
    comp.puFileData=dataDir+"/puProfile_Data12.root"
    comp.efficiency = eff2012

#for comp in dataSamplesAll:
#    comp.splitFactor = 1000
#    comp.isMC = False
#    comp.isData = True

if __name__ == "__main__":
   import sys
   if "test" in sys.argv:
       from CMGTools.TTHAnalysis.samples.ComponentCreator import testSamples
       testSamples(mcSamples)


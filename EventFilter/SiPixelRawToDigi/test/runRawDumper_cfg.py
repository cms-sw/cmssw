#
import FWCore.ParameterSet.Config as cms

process = cms.Process("d")

import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
# accept if 'path_1' succeeds
process.hltfilter = hlt.hltHighLevel.clone(
# Min-Bias
#    HLTPaths = ['HLT_Physics_v*'],
#    HLTPaths = ['HLT_Random_v*'],
#    HLTPaths = ['HLT_ZeroBias*'],
#    HLTPaths = ['HLT_L1Tech54_ZeroBias*'],
# Commissioning:
#    HLTPaths = ['HLT_L1_Interbunch_BSC_v*'],
#    HLTPaths = ['HLT_L1_PreCollisions_v1'],
#    HLTPaths = ['HLT_BeamGas_BSC_v*'],
#    HLTPaths = ['HLT_BeamGas_HF_v*'],
# LumiPixels
#    HLTPaths = ['AlCa_LumiPixels_Random_v*'],
#    HLTPaths = ['AlCa_LumiPixels_ZeroBias_v*'],
#    HLTPaths = ['AlCa_LumiPixels_v*'],
    
# examples
#    HLTPaths = ['p*'],
#    HLTPaths = ['path_?'],
    andOr = True,  # False = and, True=or
    throw = False
    )

# process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    ),
    debugModules = cms.untracked.vstring('dumper')
)
#process.MessageLogger.cerr.FwkReport.reportEvery = 1
#process.MessageLogger.cerr.threshold = 'Debug'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(                          
#    "file:../../../../../CMSSW_7_1_3/src/DPGAnalysis-SiPixelTools/HitAnalyzer/test/raw.root"
#    "file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/raw/raw2.root"
  "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/00016516-0587-E111-9884-003048D2BBF0.root",
##  "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/022D9994-1087-E111-8772-001D09F28D4A.root",
##  "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/044649F6-0B87-E111-AE83-0025B32036D2.root",
##  "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/081DAD76-0B87-E111-90EB-0030486730C6.root",
##  "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/08F21A36-0E87-E111-AB9A-00237DDBE0E2.root",
##  "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/0E9F4CD7-0587-E111-BEC8-BCAEC53296F8.root",
##  "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/106D4EE8-0987-E111-8D35-003048673374.root",
##  "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/18C4EEBC-0887-E111-A29F-5404A63886A8.root",
##  "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/1A6AF035-0E87-E111-A2B1-003048F117EA.root",
## "/store/data/Run2012A/AlCaLumiPixels/RAW/v1/000/191/271/1A6AF035-0E87-E111-A2B1-003048F117EA.root",

# fill 2576
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/16131715-E893-E111-8CDB-001D09F27003.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/1A3B279E-EB93-E111-A602-001D09F23C73.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/447ADDBD-ED93-E111-B418-003048D2C0F2.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/48BEF053-D493-E111-9043-001D09F24D4E.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/54E19592-E493-E111-90CC-5404A640A63D.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/5E0224A2-DF93-E111-9463-001D09F2932B.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/648BEF2C-D793-E111-8AA5-5404A63886B0.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/64F326FE-D993-E111-83E6-001D09F27003.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/78B5357F-DD93-E111-BD57-BCAEC53296F8.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/8AA4B112-E193-E111-84B2-003048D373F6.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RAW/v1/000/193/092/9CC6D5F9-F193-E111-AD38-003048D3750A.root",

#  "rfio:/castor/cern.ch/cms/store/data/Run2012B/Cosmics/RAW/v1/000/194/984/22BFBCD6-9CA6-E111-BB20-001D09F2932B.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012B/Commissioning/RAW/v1/000/194/984/982B2FEB-9CA6-E111-86C1-003048D37524.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012B/MinimumBias/RAW/v1/000/194/984/24265DEB-9CA6-E111-9F8D-003048F118AC.root",

# fill 2663
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/02FB3077-53A6-E111-934B-BCAEC5329700.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/0456F6A3-2CA6-E111-A084-5404A63886B6.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/06C977BC-5EA6-E111-B4B7-0025901D5C86.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/06EBF815-2BA6-E111-B535-BCAEC518FF54.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/0EAA7493-1FA6-E111-A57F-5404A63886A2.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/140F5365-1DA6-E111-9AC9-003048F024FE.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/14DCCFE3-59A6-E111-8576-0025901D5C86.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/18316693-1FA6-E111-9C7B-BCAEC518FF50.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/28ECC3D6-30A6-E111-82B3-003048F117EC.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/2A34EBB3-3CA6-E111-8E6F-0025901D626C.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/3039EE8E-3FA6-E111-8C4C-5404A63886B4.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/344DC8ED-1BA6-E111-BC0C-5404A63886A0.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/42303B27-65A6-E111-950F-5404A63886BE.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/447017D0-3EA6-E111-A972-003048D2BE08.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/44EB6113-18A6-E111-8205-5404A63886AE.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/4CF54C9F-5CA6-E111-A59F-BCAEC5364C42.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/5275E88D-1AA6-E111-9E2A-001D09F29524.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/52A3E7CF-23A6-E111-8344-0025B32036D2.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/686E41F7-34A6-E111-95A6-003048F1C420.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/6A87F431-4AA6-E111-A93E-E0CB4E4408E3.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/76167738-71A6-E111-B7C7-BCAEC532971C.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/78171AA2-25A6-E111-A8F5-001D09F23F2A.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/7819B206-63A6-E111-81AD-003048678098.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/7C2DE175-16A6-E111-BF07-003048D2C174.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/7C318BE8-4FA6-E111-8071-BCAEC5329719.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/7CD74957-3BA6-E111-A2A8-5404A63886B0.root ",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/8224362D-30A6-E111-95B3-002481E94C7E.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/881A1431-2FA6-E111-9036-BCAEC5364C42.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/8A3FD822-43A6-E111-9245-003048D2C0F2.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/92A95962-4CA6-E111-BD4B-BCAEC518FF8F.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/9AE494B7-14A6-E111-9A2A-003048F024DA.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/A609B9AD-2DA6-E111-AD0F-5404A63886A2.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/A89F0B90-15A6-E111-862D-001D09F23A20.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/B0B9BDAB-57A6-E111-AEB6-0025901D6268.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/B4374C02-19A6-E111-B8C2-002481E0D83E.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/B8A28509-37A6-E111-88EF-5404A638869E.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/B8CAB517-12A6-E111-9934-E0CB4E55365D.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/BC11AA2E-39A6-E111-B74C-BCAEC5364CED.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/BE3CB835-56A6-E111-8672-BCAEC518FF76.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/CA629788-33A6-E111-9B03-003048F024DE.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/CE83E9B0-20A6-E111-90F2-BCAEC518FF30.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/D4702F14-48A6-E111-A1CA-00215AEDFD74.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/D848E8B2-46A6-E111-A489-0025901D623C.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/D8A9AD6E-32A6-E111-9BA8-0025901D5D7E.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/D8BAD72A-27A6-E111-9F6C-003048CF99BA.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/DE875A32-1AA6-E111-9EDB-0025B324400C.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/DEDD4FAD-35A6-E111-A8F6-0025901D5DF4.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/E0AA8E53-22A6-E111-9ED2-00215AEDFD98.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/E2603AAF-44A6-E111-8241-003048F117B6.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/E279E4B7-28A6-E111-BB00-5404A638869C.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/E2B9C710-4DA6-E111-80F8-BCAEC518FF80.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/E464B525-52A6-E111-A94C-0025901D5D80.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/EC2D6375-38A6-E111-BB91-0025901D5E10.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/F2E213FA-60A6-E111-9498-001D09F27003.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/F4BB01C1-12A6-E111-A8F4-001D09F25267.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/FE1AE202-41A6-E111-935C-5404A63886C1.root",
##      "/store/data/Run2012B/MinimumBias/RAW/v1/000/194/912/FEBD36C8-6FA6-E111-B0D6-001D09F295A1.root",

# fill 2670
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/36756A9E-60A8-E111-95EF-003048F118C4.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/4EAF06E6-61A8-E111-93C4-5404A63886B4.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/7686768E-5EA8-E111-97E8-BCAEC518FF54.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/8855F6A4-73A8-E111-A8AB-003048F118AA.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/96997F92-5FA8-E111-AE0D-485B3962633D.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/9E65C1C4-5DA8-E111-B4F9-5404A63886CB.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/A628838E-61A8-E111-B1FF-BCAEC5329717.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/B6FFD156-63A8-E111-8F59-001D09F2915A.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/BE58048F-5BA8-E111-B1A3-BCAEC518FF5F.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/E0E73795-61A8-E111-807E-5404A63886C3.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/099/F2F398F9-5AA8-E111-8BD7-5404A63886EF.root",

# fill 2671
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/2A9BEBB1-7FA8-E111-BF62-BCAEC518FF8D.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/504732AA-87A8-E111-8D5C-001D09F253C0.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/50FA4C8B-83A8-E111-9046-5404A63886E6.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/68E24560-D3A8-E111-B322-5404A6388698.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/72085BB5-89A8-E111-B078-BCAEC5329717.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/92B5DF22-89A8-E111-AD1C-001D09F290CE.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/98966276-80A8-E111-BE10-BCAEC5364C93.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/A016F8DE-86A8-E111-84EE-E0CB4E55365C.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/ACB04E2B-84A8-E111-921E-0025901D624A.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/BE659A2E-82A8-E111-8C93-5404A63886AD.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/E8A3133E-87A8-E111-B525-BCAEC5364C4C.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/F6733579-7EA8-E111-B97C-5404A63886C1.root",
##    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/109/FC8BD284-85A8-E111-ADEA-003048D3C982.root",


# fill 2712
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/F0C3F680-F9B0-E111-9317-5404A638869E.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/DACC63F4-16B1-E111-80BB-001D09F297EF.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/D647D869-EDB0-E111-BD2F-BCAEC518FF67.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/D43C1410-18B1-E111-BE66-E0CB4E553651.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/CACFE8BE-15B1-E111-9EB1-BCAEC5329719.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/C8393952-FCB0-E111-9D6F-5404A63886B2.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/C2AD82E4-EEB0-E111-9F8A-001D09F241F0.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/B6AC16BD-F8B0-E111-838D-0025901D5C86.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/AE671928-0FB1-E111-880E-0025901D5D7E.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/A47B173A-F0B0-E111-B39A-BCAEC5364CED.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/9E0023CF-F3B0-E111-819B-BCAEC5329708.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/94A50F5D-01B1-E111-92E9-BCAEC532970D.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/8A5B2E56-0BB1-E111-A1AD-0025901D5DF4.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/7E295219-0DB1-E111-8C21-5404A63886C1.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/7A8C4860-11B1-E111-9C3C-003048F117EA.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/722A9DEE-1AB1-E111-8EEA-001D09F2525D.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/6ECA1C6A-F2B0-E111-8495-485B39897227.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/68D6CFAF-07B1-E111-AAB9-5404A63886AF.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/68A6A873-EBB0-E111-B1B3-485B39897227.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/5EE21EF8-F0B0-E111-AC2E-5404A6388698.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/4C366872-EBB0-E111-80D5-BCAEC518FF5F.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/48D71235-F5B0-E111-BCD4-5404A63886A0.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/46726EEC-E9B0-E111-99F3-5404A63886B4.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/42DF2CAA-09B1-E111-B753-0025901D5C88.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/42812E18-F3B0-E111-A117-BCAEC518FF62.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/3A1B4329-FFB0-E111-B8B9-BCAEC532971B.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/3444D555-13B1-E111-AAE3-BCAEC518FF8D.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/3090B05D-F7B0-E111-BF59-5404A640A642.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/2E71EC2B-04B1-E111-912F-BCAEC5329713.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/288E6621-ECB0-E111-AC4A-5404A638869E.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/1E3D187A-14B1-E111-9EE4-5404A63886B2.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/1E144854-06B1-E111-A05B-BCAEC518FF7C.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/1CE271CB-FDB0-E111-98A2-003048D2C020.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/1C907D73-19B1-E111-9EF7-003048D2BED6.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/1C6907C9-47B1-E111-A398-003048F1183E.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/18912BC6-02B1-E111-BB2D-5404A640A639.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/08AF31A0-F6B0-E111-8976-003048F118C2.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/068F45EB-FAB0-E111-A522-5404A63886A0.root",
## "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/774/006ED8A1-00B1-E111-BD0F-0025901D629C.root",


# fill 2713
#    "/store/data/Run2012B/MinimumBias/RAW/v1/000/195/841/7884E5C6-7BB1-E111-B2AB-0025901D625A.root",

# fill 2825 high PU
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/FEC78C8C-72CA-E111-B6CB-BCAEC5329708.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/F0C83922-74CA-E111-909B-003048673374.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/EE067D66-74CA-E111-A6A1-003048CF9B28.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/A064B44B-75CA-E111-81B5-001D09F24DA8.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/9EA624E6-75CA-E111-8B11-5404A63886B4.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/8E8E728C-72CA-E111-A5A2-5404A63886C5.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/80BCF960-74CA-E111-AF2A-5404A63886C6.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/7E49918B-72CA-E111-A7C3-BCAEC518FF8D.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/7AF09F1E-74CA-E111-8654-003048D2BF1C.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/76862D8D-72CA-E111-95E8-5404A63886EE.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/66ACE6D3-70CA-E111-A1A9-003048F1C832.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/58A2ABD6-70CA-E111-9327-0030486780E6.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/408C37C9-70CA-E111-8A54-003048CF9B28.root",
## "/store/data/Run2012C/ZeroBias/RAW/v1/000/198/609/36D31C2B-74CA-E111-9A87-003048D37560.root",

# RAW dtaa not anymor on EOS
#  "rfio:/castor/cern.ch/cms/store/data/Run2012C/MinimumBias/RAW/v1/000/201/657/3601D580-41EE-E111-B55A-0025901D5DF4.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012C/MinimumBias/RAW/v1/000/201/657/96B51DAE-3EEE-E111-95ED-0019B9F72CE5.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012C/MinimumBias/RAW/v1/000/201/657/E2C43972-3AEE-E111-9D26-003048F024FE.root",

#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/2EF61B7D-F216-E211-98C3-001D09F28D54.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/6825CA93-0017-E211-8B46-001D09F25267.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/6C8D4EB2-F116-E211-A7CD-0019B9F70468.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/96B95E6C-FE16-E211-823E-001D09F295FB.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/AEF11AB6-FD16-E211-AA6B-001D09F24399.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/CA8B57D2-F316-E211-8DB2-003048D373F6.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/DA71CB04-FD16-E211-8254-0019B9F4A1D7.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/F859580F-F116-E211-B759-485B39897227.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/FC46615A-F716-E211-924F-001D09F276CF.root",


#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/718/02285033-FD1B-E211-8F74-001D09F295FB.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/718/08787828-EE1B-E211-B680-0025901D5D78.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/718/1285A278-E41B-E211-BD9A-0019B9F72F97.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/718/1289B6A0-F91B-E211-BC25-001D09F24682.root",

#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/208/686/A88F66A0-393F-E211-9287-002481E0D524.root",
 
    )

)

#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('191271:55-191271:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('191718:30-191718:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('194912:52-194912:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('194912:52-194912:330 ')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('195099:61-195099:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('195109:85-195109:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('195841:73-195841:100','195841:116-195841:143')
# 195774 OK from LS=0
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('198609:47-198609:112')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('201657:77-201657:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('205217:0-205217:323')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('205718:49-205718:734')

process.d = cms.EDAnalyzer("SiPixelRawDumper", 
    Timing = cms.untracked.bool(False),
    IncludeErrors = cms.untracked.bool(True),
#   In 2012, label = rawDataCollector, extension = _LHC                                
#    InputLabel = cms.untracked.string('rawDataCollector'),
# for MC
    InputLabel = cms.untracked.string('siPixelRawData'),
#   For PixelLumi stream                           
#    InputLabel = cms.untracked.string('hltFEDSelectorLumiPixels'),
# old
#    InputLabel = cms.untracked.string('siPixelRawData'),
#    InputLabel = cms.untracked.string('source'),
    CheckPixelOrder = cms.untracked.bool(False),
# 0 - nothing, 1 - error , 2- data, 3-headers, 4-hex
    Verbosity = cms.untracked.int32(1),
# threshold, print fed/channel num of errors if tot_errors > events * PrintThreshold, default 0,001 
    PrintThreshold = cms.untracked.double(0.001)
)

# process.p = cms.Path(process.hltfilter*process.d)
process.p = cms.Path(process.d)

# process.ep = cms.EndPath(process.out)



import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_0_15_10TeV_GenJets_800Kevts_ptHatFiltered.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_470_600_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_1400_1800_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_1000_1400_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_1800_2200_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_2200_2600_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_300_380_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_2600_3000_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_230_300_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_3000_3500_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_170_230_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_120_170_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_800_1000_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_600_800_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_380_470_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_20_30_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_3500_5000_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_80_120_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_15_20_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_50_80_10TeV_GenJets_800Kevts.root',
'dcap://cmsgridftp.fnal.gov:24125/pnfs/fnal.gov/usr/cms/WAX/resilient/rharris/MC/QCD_2_1_8/PYTHIA6_QCDpt_30_50_10TeV_GenJets_800Kevts.root')
)

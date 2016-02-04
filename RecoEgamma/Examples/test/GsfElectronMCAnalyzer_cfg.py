
import httplib, urllib, urllib2, types, string, os, sys
import FWCore.ParameterSet.Config as cms

code_release = "CMSSW_3_4_0_pre5"
data_sample = "RelValSingleElectronPt35"
data_set_like = "*MC_3XY_V14-v1*-RECO*"

process = cms.Process("readelectrons")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring(),
    secondaryFileNames = cms.untracked.vstring(),
)

from RecoEgamma.Examples.mcAnalyzerStdBiningParameters_cff import *
from RecoEgamma.Examples.mcAnalyzerFineBiningParameters_cff import *

process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronMCAnalyzer",
  electronCollection = cms.InputTag("gsfElectrons"),
  mcTruthCollection = cms.InputTag("genParticles"),
  readAOD = cms.bool(False),
  outputFile = cms.string(data_sample+".GsfElectronMCAnalyzer.root"),
  MaxPt = cms.double(100.0),
  DeltaR = cms.double(0.05),
  MatchingID = cms.vint32(11,-11),
  MatchingMotherID = cms.vint32(23,24,-24,32),
  MaxAbsEta = cms.double(2.5),
  HistosConfigurationMC = cms.PSet(
    mcAnalyzerStdBiningParameters
    #mcAnalyzerFineBiningParameters
  )
)

process.p = cms.Path(process.gsfElectronAnalysis)

#==============================================
# the code below get the list of files from DBS
#============================================== 

url = "https://cmsweb.cern.ch:443/dbs_discovery/aSearch"
input = "find file where release = " + code_release
input = input + " and primds = " + data_sample
input = input + " and dataset like " + data_set_like
final_input = urllib.quote(input) ;
params  = {
  'dbsInst':'cms_dbs_prod_global',
  'html':0,'caseSensitive':'on','_idx':0,'pagerStep':-1,
  'userInput':final_input,
  'xml':0,'details':0,'cff':0,'method':'dbsapi'
}
data = urllib.urlencode(params,doseq=True)
headers = {
  'User-Agent':'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)',
  'Accept':'text/plain'
}
req  = urllib2.Request(url, data, headers)
for line in urllib2.urlopen(req).read().split("\n"):
  if line != "" and line[0] =="/":
    process.source.fileNames.append(line)



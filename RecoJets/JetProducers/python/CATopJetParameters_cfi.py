import FWCore.ParameterSet.Config as cms

# Cambridge-Aachen top jet producer parameters
# $Id
CATopJetParameters = cms.PSet(
    jetCollInstanceName = cms.string("caTopSubJets"),	# subjet collection
    verbose = cms.bool(False),          
    algorithm = cms.int32(1),               			# 0 = KT, 1 = CA, 2 = anti-KT
    useAdjacency = cms.int32(2),         				# veto adjacent subjets
														#  0 = no adjacency
														#  1 = deltar adjacency 
                                                        #  2 = modified adjacency
                                                        #  3 = calotower neirest neigbor based adjacency (untested)
    centralEtaCut = cms.double(2.5),        			# eta for defining "central" jets                                     
    sumEtBins = cms.vdouble(0,1600,2600),          		# sumEt bins over which cuts vary. vector={bin 0 lower bound, bin 1 lower bound, ...} 
    rBins = cms.vdouble(0.8,0.8,0.8),           		# Jet distance paramter R. R values depend on sumEt bins.
    ptFracBins = cms.vdouble(0.05,0.05,0.05),    		# minimum fraction of central jet pt for subjets (deltap)
	deltarBins = cms.vdouble(0.19,0.19,0.19),           # Applicable only if useAdjacency=1. deltar adjacency values for each sumEtBin
	nCellBins = cms.vdouble(1.9,1.9,1.9),           	# Applicable only if useAdjacency=3. number of cells apart for two subjets to be considered "independent"

#NOT USED:
    useMaxTower = cms.bool(False),          			# use max tower in adjacency criterion, otherwise use centroid - NOT USED
    sumEtEtaCut = cms.double(3.0),          			# eta for event SumEt - NOT USED                                                 
    etFrac = cms.double(0.7),               			# fraction of event sumEt / 2 for a jet to be considered "hard" - NOT USED
    debugLevel = cms.untracked.int32(0)     			# debug level
)


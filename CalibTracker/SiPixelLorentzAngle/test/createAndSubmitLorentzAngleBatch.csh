#!/bin/tcsh -f


echo "using name: " $1 
echo "start number: " $2
echo "end number: " $3

##iCounter is the variable counting the jobs
set iCounter=${2}
while ( $iCounter < $3 )
  echo "creating job " $iCounter

## maybe multiply number of events to be skipped by job number
set nEvents = 1000
set nSkipEvents = $nEvents
@ nSkipEvents*=${iCounter}

  echo "skipping events: " $nSkipEvents


## creating shell script to be submitted
cat > ${1}_${iCounter}.csh <<EOI
#!/bin/tcsh -f

scramv1 project CMSSW CMSSW_2_0_6
cd CMSSW_2_0_6
cd src

eval \`scramv1 run -csh\`

cvs co -r V00-03-09 CalibTracker/SiPixelLorentzAngle
cvs co -r V07-06-06-01 IOPool/Input

cd CalibTracker/SiPixelLorentzAngle
scramv1 b

###creating config file
cat > config.cfg <<EOF

process LorentzAngle = {

// service = MessageLogger {
//     untracked vstring destinations = { "/home/wilke/debug.txt"
//                                      , "/home/wilke/errors.txt"
//                                      }
//     untracked PSet debug.txt = { untracked string threshold = "DEBUG"    } 
//     untracked vstring debugModules =  { "read" 
// 			              }
//   }

untracked PSet maxEvents = {untracked int32 input = $nEvents}

source = PoolSource { 
	untracked vstring fileNames = {
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0021/00E04B2E-261E-DD11-A289-001D09F251CC.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0021/C0CB7326-281E-DD11-9349-0019B9F72D71.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0022/5E72C867-2A1E-DD11-B475-0019B9F72BFF.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0022/6A01ADF8-321E-DD11-87D1-001617DBD288.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0022/76892818-401E-DD11-AA30-000423D9870C.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0022/8650C21A-361E-DD11-911E-001617C3B70E.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0022/8A0A6C53-4F1E-DD11-A68B-001617DBD5B2.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0022/A802A049-2D1E-DD11-8980-000423D992A4.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0022/B60EE35E-1820-DD11-868F-000423D9880C.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0022/C401ABB6-AD1E-DD11-A979-000423D9853C.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0022/E66507FE-381E-DD11-B6C4-000423D98DC4.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0025/26333C78-9320-DD11-AA9E-001D09F282F5.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0025/3098C8B5-9120-DD11-A99C-001D09F28F25.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0025/8CB1D3A2-9220-DD11-8251-001D09F24259.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/004612F7-A120-DD11-BABD-000423D6B48C.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/06F52CBC-A320-DD11-AF58-000423D98FBC.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/1857A4B7-9E20-DD11-BA3D-001D09F24DA8.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/2AE8F245-9C20-DD11-B681-001D09F29524.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/2CC8132C-9F20-DD11-B1B5-001617C3B79A.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/660C7AB1-9E20-DD11-807B-000423D98A44.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/6AD44739-A520-DD11-A037-001D09F25041.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/84D6CE6E-A420-DD11-9E7D-001D09F2906A.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/8835CAEF-9720-DD11-802A-001617E30D38.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/A008DB04-9B20-DD11-9825-001D09F23A4D.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/AC07E2DE-9F20-DD11-92CC-0019B9F704D6.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/B04B5EF0-9720-DD11-A315-000423D98B6C.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/F82D66A9-9D20-DD11-B6BC-001617C3B710.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0026/FC84AC32-9620-DD11-BD94-000423D6A6F4.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/20F983F7-AC20-DD11-AB6F-001617C3B78C.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/50500F6E-B020-DD11-A452-001617C3B65A.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/7073B3F2-A820-DD11-9052-000423D6B5C4.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/9C50A315-AB20-DD11-9216-000423D99614.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/B0F3BEE2-A620-DD11-BBE6-000423D6B42C.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/B2F3CD80-AC20-DD11-8DEF-001617C3B73A.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/CC1836D4-A720-DD11-8942-001617E30D38.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/E008F5D7-A920-DD11-8152-000423D987E0.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/FA916FF2-BC20-DD11-AAD2-000423D98DB4.root',
		'/store/mc/CSA08/MuonPT5/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0027/FE6B12C1-AE20-DD11-882C-000423D992DC.root'

# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/164B91A4-1F1A-DD11-8857-001D09F2527B.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/28CD9B00-201A-DD11-9FB2-000423D60FF6.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/2E397D9D-AC1A-DD11-B915-0019DB2F3F9B.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/2ED908E7-141A-DD11-9888-000423D98AF0.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/306655C8-C51A-DD11-BF4B-0016177CA7A0.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/6202B313-861A-DD11-A0CB-000423D6BA18.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/64E239F0-CC1A-DD11-914D-000423D6B5C4.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/7A99EBCC-1C1A-DD11-85A4-001D09F28F0C.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/A28281AB-CD1A-DD11-ABF4-001617C3B73A.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/D254AF58-141A-DD11-AC5B-001617E30D52.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/D6170650-181A-DD11-86AD-001D09F2423B.root',
# 		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/DC48A5EB-B41A-DD11-B778-000423D992DC.root'
		
# 		'/store/mc/CSA08/MinBias/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0016/949FCBEA-991A-DD11-A596-000423D9880C.root'
	}
	untracked uint32 skipEvents = $nSkipEvents
	untracked bool useCSA08Kludge = true
}



# magnetic field
include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"
 
# tracker geometry
// include "Geometry/TrackerRecoData/data/trackerRecoGeometryXML.cfi"

# tracker geometry
include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
 
# tracker numbering
include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"
 
# KFUpdatoerESProducer
include "TrackingTools/KalmanUpdators/data/KFUpdatorESProducer.cfi"
 
# Chi2MeasurementEstimatorESProducer
include "TrackingTools/KalmanUpdators/data/Chi2MeasurementEstimatorESProducer.cfi" 

# KFTrajectoryFitterESProducer
include "TrackingTools/TrackFitters/data/KFTrajectoryFitterESProducer.cfi"

# KFTrajectorySmootherESProducer
include "TrackingTools/TrackFitters/data/KFTrajectorySmootherESProducer.cfi"

# KFFittingSmootherESProducer
include "TrackingTools/TrackFitters/data/KFFittingSmootherESProducer.cfi"

# PropagatorWithMaterialESProducer
include "TrackingTools/MaterialEffects/data/MaterialPropagator.cfi"

# PropagatorWithMaterialESProducer
include "TrackingTools/MaterialEffects/data/OppositeMaterialPropagator.cfi"

# stripCPE
include "RecoLocalTracker/SiStripRecHitConverter/data/StripCPEfromTrackAngle.cfi"

# pixelCPE
include "RecoLocalTracker/SiPixelRecHits/data/PixelCPEParmError.cfi"

#TransientTrackingBuilder
include "RecoTracker/TransientTrackingRecHit/data/TransientTrackingRecHitBuilder.cfi"

// include "RecoLocalTracker/SiStripRecHitConverter/data/SiStripRecHitMatcher.cfi"

include "Configuration/StandardSequences/data/FakeConditions.cff"
//    include "RecoLocalTracker/SiPixelClusterizer/data/SiPixelClusterizer.cfi"
//    include "RecoLocalMuon/DTRecHit/data/DTParametrizedDriftAlgo_CSA07.cfi"



module read = SiPixelLorentzAngle{
// 	bool MTCCtrack= false
	string TTRHBuilder="WithTrackAngle"
	string Fitter = "KFFittingSmoother"   
  	string Propagator = "PropagatorWithMaterial"
	#what type of tracks should be used: 
//   	string src = "generalTracks"
  	string src = "globalMuons"
// 	string src = "ALCARECOTkAlMinBias"
	string fileName = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/Pixels/LorentzAngle/Results/CMSSW_2_0_6/lorentzAngle_${1}_${iCounter}.root"
	string fileNameFit	= "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/Pixels/LorentzAngle/Results/CMSSW_2_0_6/lorentzFit_${1}_${iCounter}.txt"
	int32 binsDepth	= 50
	int32 binsDrift =	60
	double ptMin = 3
	#in case of MC set this to true to save the simhits
	bool simData = false
	}
	
	module mix = MixingModule
   {
      int32 bunchspace = 25
      string Label=""
   }
	path p = {read}
}


EOF
EOI


cat >> ${1}_${iCounter}.csh <<EOI

## end of creating config file

eval \`scramv1 run -csh\`
cmsRun config.cfg >& /afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/Pixels/LorentzAngle/Results/CMSSW_2_0_6/log${1}_${iCounter}.txt

EOI

## end of shell script

## now submitting the script:

chmod u+x ${1}_${iCounter}.csh
echo "submitting file: " ${1}_${iCounter}.csh
bsub -q cmscaf ${1}_${iCounter}.csh

## this increses the iCounter variable by one
@ iCounter++
end

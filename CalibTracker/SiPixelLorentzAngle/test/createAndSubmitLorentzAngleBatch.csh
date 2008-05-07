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

cvs co -r V00-03-08 CalibTracker/SiPixelLorentzAngle
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

source = PoolSource { 
	untracked vstring fileNames = {	
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/164B91A4-1F1A-DD11-8857-001D09F2527B.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/28CD9B00-201A-DD11-9FB2-000423D60FF6.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/2E397D9D-AC1A-DD11-B915-0019DB2F3F9B.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/2ED908E7-141A-DD11-9888-000423D98AF0.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/306655C8-C51A-DD11-BF4B-0016177CA7A0.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/6202B313-861A-DD11-A0CB-000423D6BA18.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/64E239F0-CC1A-DD11-914D-000423D6B5C4.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/7A99EBCC-1C1A-DD11-85A4-001D09F28F0C.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/A28281AB-CD1A-DD11-ABF4-001617C3B73A.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/D254AF58-141A-DD11-AC5B-001617E30D52.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/D6170650-181A-DD11-86AD-001D09F2423B.root',
		'/store/relval/2008/5/4/RelVal-RelValZMM-1209251027-STARTUP_V2-3rd/0000/DC48A5EB-B41A-DD11-B778-000423D992DC.root'
		
# 		'/store/mc/CSA08/MinBias/ALCARECO/STARTUP_V2_SiPixelLorentzAngle_v1/0016/949FCBEA-991A-DD11-A596-000423D9880C.root'
	}
}

untracked PSet maxEvents = {untracked int32 input = $nEvents}
untracked PSet skipEvents = {untracked int32 input = $nSkipEvents}

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
	string fileName = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/Pixels/LorentzAngle/Results/CMSSW_2_0_6/test/lorentzAngle_${iCounter}.root"
	string fileNameFit	= "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/Pixels/LorentzAngle/Results/CMSSW_2_0_6/test/lorentzFit_${iCounter}.txt"
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
cmsRun config.cfg >& /afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/Pixels/LorentzAngle/Results/CMSSW_2_0_6/test/log${1}_${iCounter}.txt

EOI

## end of shell script

## now submitting the script:

chmod u+x ${1}_${iCounter}.csh
echo "submitting file: " ${1}_${iCounter}.csh
bsub -q cmscaf ${1}_${iCounter}.csh

## this increses the iCounter variable by one
@ iCounter++
end

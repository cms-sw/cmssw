#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "RecoParticleFlow/PFClusterTools/interface/Calibrator.h"  
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElement.h"      
#include "RecoParticleFlow/PFClusterTools/interface/LinearCalibrator.h"    
#include "RecoParticleFlow/PFClusterTools/interface/Operators.h"         
#include "RecoParticleFlow/PFClusterTools/interface/SpaceManager.h"  
#include "RecoParticleFlow/PFClusterTools/interface/ToString.h"
#include "RecoParticleFlow/PFClusterTools/interface/Deposition.h"  
#include "RecoParticleFlow/PFClusterTools/interface/DetectorElementType.h"  
#include "RecoParticleFlow/PFClusterTools/interface/PFToolsException.h"  
#include "RecoParticleFlow/PFClusterTools/interface/ParticleDeposit.h"  
#include "RecoParticleFlow/PFClusterTools/interface/SpaceVoxel.h"    
#include "RecoParticleFlow/PFClusterTools/interface/TreeUtility.h"
#include "RecoParticleFlow/PFClusterTools/interface/SingleParticleWrapper.h"
#include "RecoParticleFlow/PFClusterTools/interface/Exercises.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationResultWrapper.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationProvenance.h"
#include "RecoParticleFlow/PFClusterTools/interface/CalibrationTarget.h"
namespace { 
  namespace {
	pftools::SingleParticleWrapper spw;
	pftools::CalibrationResultWrapper crw;
  }
}

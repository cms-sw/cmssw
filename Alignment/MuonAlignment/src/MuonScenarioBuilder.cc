/** \file
 *
 *  $Date: 2011/10/12 22:13:10 $
 *  $Revision: 1.10 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

#include <string>
#include <iostream>
#include <sstream>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment

#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "Alignment/CommonAlignment/interface/Alignable.h" 
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h" 

//__________________________________________________________________________________________________
MuonScenarioBuilder::MuonScenarioBuilder( Alignable* alignable )
{

  theAlignableMuon = dynamic_cast<AlignableMuon*>( alignable );

  if ( !theAlignableMuon )
    throw cms::Exception("TypeMismatch") << "Argument is not an AlignableMuon";

}


//__________________________________________________________________________________________________
void MuonScenarioBuilder::applyScenario( const edm::ParameterSet& scenario )
{
  // Apply the scenario to all main components of Muon.
  theScenario = scenario;
  theModifierCounter = 0;

  // Seed is set at top-level, and is mandatory
  if ( this->hasParameter_( "seed", theScenario ) )
	theModifier.setSeed( static_cast<long>(theScenario.getParameter<int>("seed")) );
  else
	throw cms::Exception("BadConfig") << "No generator seed defined!";  



  // DT Barrel
  std::vector<Alignable*> dtBarrel = theAlignableMuon->DTBarrel();
  this->decodeMovements_( theScenario, dtBarrel, "DTBarrel" );
  // CSC Endcap
  std::vector<Alignable*> cscEndcaps = theAlignableMuon->CSCEndcaps();
  this->decodeMovements_( theScenario, cscEndcaps, "CSCEndcap" );

  this->moveDTSectors(theScenario);
  this->moveCSCSectors(theScenario);
  this->moveMuon(theScenario);
  
  edm::LogInfo("TrackerScenarioBuilder") 
	<< "Applied modifications to " << theModifierCounter << " alignables";
}



align::Scalars MuonScenarioBuilder::extractParameters(const edm::ParameterSet& pSet, const char *blockId)
{
  double scale_ = 0, scaleError_ = 0, phiX_ = 0, phiY_ = 0, phiZ_ = 0;
  double dX_ = 0, dY_ = 0, dZ_ = 0;
  std::string distribution_;
  std::ostringstream error;
  edm::ParameterSet Parameters = this->getParameterSet_((std::string)blockId, pSet);
  std::vector<std::string> parameterNames = Parameters.getParameterNames();
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin(); iParam != parameterNames.end(); iParam++ ) {
    if ( (*iParam) == "scale" )    scale_ = Parameters.getParameter<double>( *iParam );
    else if ( (*iParam) == "distribution" ) distribution_ = Parameters.getParameter<std::string>( *iParam );
    else if ( (*iParam) == "scaleError" ) scaleError_ = Parameters.getParameter<double>( *iParam );
    else if ( (*iParam) == "phiX" )     phiX_     = Parameters.getParameter<double>( *iParam );
    else if ( (*iParam) == "phiY" )     phiY_     = Parameters.getParameter<double>( *iParam );
    else if ( (*iParam) == "phiZ" )     phiZ_     = Parameters.getParameter<double>( *iParam );
    else if ( (*iParam) == "dX" )       dX_       = Parameters.getParameter<double>( *iParam );
    else if ( (*iParam) == "dY" )       dY_       = Parameters.getParameter<double>( *iParam );
    else if ( (*iParam) == "dZ" )       dZ_       = Parameters.getParameter<double>( *iParam );
    else if ( Parameters.retrieve( *iParam ).typeCode() != 'P' )
      { // Add unknown parameter to list
	if ( !error.str().length() ) error << "Unknown parameter name(s): ";
	error << " " << *iParam;
      }
  }
  align::Scalars param;
  param.push_back(scale_); param.push_back(scaleError_);
  param.push_back(phiX_); param.push_back(phiY_);
  param.push_back(phiZ_); param.push_back(dX_);
  param.push_back(dY_); param.push_back(dZ_);
  if( distribution_ == "gaussian" )
    param.push_back(0);
  else if ( distribution_ == "flat" )
    param.push_back(1);
  else if ( distribution_ == "fix" )
    param.push_back(2);
  
  return param;
}

//_____________________________________________________________________________________________________
void MuonScenarioBuilder::moveDTSectors(const edm::ParameterSet& pSet)
{
  std::vector<Alignable *> DTchambers = theAlignableMuon->DTChambers();
  //Take parameters
  align::Scalars param = this->extractParameters(pSet, "DTSectors");
  double scale_ = param[0]; double scaleError_ = param[1];
  double phiX_ = param[2]; double phiY_ = param[3]; double phiZ_ = param[4];
  double dX_ = param[5]; double dY_ = param[6]; double dZ_ = param[7];
  double dist_ = param[8];

  double dx = scale_*dX_; double dy = scale_*dY_; double dz = scale_*dZ_;
  double phix = scale_*phiX_; double phiy = scale_*phiY_; double phiz = scale_*phiZ_;
  double errorx = scaleError_*dX_; double errory = scaleError_*dY_; double errorz = scaleError_*dZ_;
  double errorphix = scaleError_*phiX_; double errorphiy = scaleError_*phiY_; double errorphiz = scaleError_*phiZ_;
  align::Scalars errorDisp;
  errorDisp.push_back(errorx); errorDisp.push_back(errory); errorDisp.push_back(errorz);
  align::Scalars errorRotation;
  errorRotation.push_back(errorphix); errorRotation.push_back(errorphiy); errorRotation.push_back(errorphiz);
 
  int index[5][4][14];
  int counter = 0;
  //Create and index for the chambers in the Alignable vector
  for(std::vector<Alignable *>::iterator iter = DTchambers.begin(); iter != DTchambers.end(); ++iter) {
    DTChamberId myId((*iter)->geomDetId().rawId());
    index[myId.wheel()+2][myId.station()-1][myId.sector()-1] = counter;
    counter++;
  }
  for(int wheel = 0; wheel < 5; wheel++) {
    for(int sector = 0; sector < 12; sector++) {
      align::Scalars disp;
      align::Scalars rotation;
      if( dist_ == 0 ) {
        const std::vector<float> disp_ = theMuonModifier.gaussianRandomVector(dx, dy, dz);
        const std::vector<float> rotation_ = theMuonModifier.gaussianRandomVector(phix, phiy, phiz);
        disp.push_back(disp_[0]); disp.push_back(disp_[1]); disp.push_back(disp_[2]);
        rotation.push_back(rotation_[0]); rotation.push_back(rotation_[1]); rotation.push_back(rotation_[2]);
      } else if (dist_ == 1) {
        const std::vector<float> disp_ = theMuonModifier.flatRandomVector(dx, dy, dz);
        const std::vector<float> rotation_ = theMuonModifier.flatRandomVector(phix, phiy, phiz);
        disp.push_back(disp_[0]); disp.push_back(disp_[1]); disp.push_back(disp_[2]);
        rotation.push_back(rotation_[0]); rotation.push_back(rotation_[1]); rotation.push_back(rotation_[2]);
      } else {
        disp.push_back(dx); disp.push_back(dy); disp.push_back(dz);
        rotation.push_back(phix); rotation.push_back(phiy); rotation.push_back(phiz);
      }
      for(int station = 0; station < 4; station++) {
        Alignable *myAlign = DTchambers.at(index[wheel][station][sector]);
        this->moveChamberInSector(myAlign, disp, rotation, errorDisp, errorRotation);
        if(sector == 3 && station == 3) {
	  Alignable *myAlignD = DTchambers.at(index[wheel][station][12]);
          this->moveChamberInSector(myAlignD, disp, rotation, errorDisp, errorRotation);
	} else if(sector == 9 && station == 3) {
	  Alignable *myAlignD = DTchambers.at(index[wheel][station][13]);
          this->moveChamberInSector(myAlignD, disp, rotation, errorDisp, errorRotation);
        }
      }
    }
  } 
}


//______________________________________________________________________________________________________
void MuonScenarioBuilder::moveCSCSectors(const edm::ParameterSet& pSet)
{
  std::vector<Alignable *> CSCchambers = theAlignableMuon->CSCChambers();
  //Take Parameters
  align::Scalars param = this->extractParameters(pSet, "CSCSectors");
  double scale_ = param[0]; double scaleError_ = param[1];
  double phiX_ = param[2]; double phiY_ = param[3]; double phiZ_ = param[4];
  double dX_ = param[5]; double dY_ = param[6]; double dZ_ = param[7];
  double dist_ = param[8];
  
  double dx = scale_*dX_; double dy = scale_*dY_; double dz = scale_*dZ_;
  double phix = scale_*phiX_; double phiy = scale_*phiY_; double phiz = scale_*phiZ_;
  double errorx = scaleError_*dX_; double errory = scaleError_*dY_; double errorz = scaleError_*dZ_;
  double errorphix = scaleError_*phiX_; double errorphiy = scaleError_*phiY_; double errorphiz = scaleError_*phiZ_;
  align::Scalars errorDisp;
  errorDisp.push_back(errorx); errorDisp.push_back(errory); errorDisp.push_back(errorz);
  align::Scalars errorRotation;
  errorRotation.push_back(errorphix); errorRotation.push_back(errorphiy); errorRotation.push_back(errorphiz);
  
  int index[2][4][4][36];
  int sector_index[2][4][4][36];
  int counter = 0;
  //Create an index for the chambers in the alignable vector
  for(std::vector<Alignable *>::iterator iter = CSCchambers.begin(); iter != CSCchambers.end(); ++iter) {
    CSCDetId myId((*iter)->geomDetId().rawId());
    index[myId.endcap()-1][myId.station()-1][myId.ring()-1][myId.chamber()-1] = counter;
    sector_index[myId.endcap()-1][myId.station()-1][myId.ring()-1][myId.chamber()-1] = CSCTriggerNumbering::sectorFromTriggerLabels(CSCTriggerNumbering::triggerSectorFromLabels(myId),CSCTriggerNumbering::triggerSubSectorFromLabels(myId) , myId.station());
    counter++;
  }
  for(int endcap = 0; endcap < 2; endcap++) {
    for(int ring = 0; ring < 2; ring++) {
      for(int sector = 1; sector < 7; sector++) {
        align::Scalars disp;
        align::Scalars rotation;
        if( dist_ == 0 ) {
          const std::vector<float> disp_ = theMuonModifier.gaussianRandomVector(dx, dy, dz);
          const std::vector<float> rotation_ = theMuonModifier.gaussianRandomVector(phix, phiy, phiz);
          disp.push_back(disp_[0]); disp.push_back(disp_[1]); disp.push_back(disp_[2]);
          rotation.push_back(rotation_[0]); rotation.push_back(rotation_[1]); rotation.push_back(rotation_[2]);
        } else if (dist_ == 1) {
          const std::vector<float> disp_ = theMuonModifier.flatRandomVector(dx, dy, dz);
          const std::vector<float> rotation_ = theMuonModifier.flatRandomVector(phix, phiy, phiz);
          disp.push_back(disp_[0]); disp.push_back(disp_[1]); disp.push_back(disp_[2]);
          rotation.push_back(rotation_[0]); rotation.push_back(rotation_[1]); rotation.push_back(rotation_[2]);
        } else {
          disp.push_back(dx); disp.push_back(dy); disp.push_back(dz);
          rotation.push_back(phix); rotation.push_back(phiy); rotation.push_back(phiz);
        }
        //Different cases are considered in order to fit endcap geometry
	for(int station = 0; station < 4; station++) {
	  if(station == 0) {
	    int r_ring[2];
	    if(ring == 0) {
	      r_ring[0] = 0; r_ring[1] = 3;
	    } else {
	      r_ring[0] = 1; r_ring[1] = 2;
	    }
	    for(int r_counter = 0; r_counter < 2; r_counter++) {
	      for(int chamber = 0; chamber < 36; chamber++) {
		if(sector == (sector_index[endcap][station][r_ring[r_counter]][chamber]+1)/2) {
		  Alignable *myAlign = CSCchambers.at(index[endcap][station][r_ring[r_counter]][chamber]);
                  this->moveChamberInSector(myAlign, disp, rotation, errorDisp, errorRotation);
		}
	      }
	    }
	  } else if(station == 3 && ring == 1) {
	    continue;
	  } else {
	    for(int chamber = 0; chamber < 36; chamber++) {
	      if(ring == 0 && chamber > 17) continue;
	      if(sector == sector_index[endcap][station][ring][chamber]) {
		Alignable *myAlign = CSCchambers.at(index[endcap][station][ring][chamber]);
                this->moveChamberInSector(myAlign, disp, rotation, errorDisp, errorRotation);
	      }
	    }
	  }
	}
      }
    }
  }
}


//______________________________________________________________________________________________________
void MuonScenarioBuilder::moveMuon(const edm::ParameterSet& pSet)
{
  std::vector<Alignable *> DTbarrel = theAlignableMuon->DTBarrel();	
  std::vector<Alignable *> CSCendcaps = theAlignableMuon->CSCEndcaps();  
  //Take Parameters
  align::Scalars param = this->extractParameters(pSet, "Muon");
  double scale_ = param[0]; double scaleError_ = param[1];
  double phiX_ = param[2]; double phiY_ = param[3]; double phiZ_ = param[4];
  double dX_ = param[5]; double dY_ = param[6]; double dZ_ = param[7];
  double dist_ = param[8]; 
  double dx = scale_*dX_; double dy = scale_*dY_; double dz = scale_*dZ_;
  double phix = scale_*phiX_; double phiy = scale_*phiY_; double phiz = scale_*phiZ_;
  double errorx = scaleError_*dX_; double errory = scaleError_*dY_; double errorz = scaleError_*dZ_;
  double errorphix = scaleError_*phiX_; double errorphiy = scaleError_*phiY_; double errorphiz = scaleError_*phiZ_;
  //Create an index for the chambers in the alignable vector
  align::Scalars disp;
  align::Scalars rotation;
  if( dist_ == 0 ) {
    const std::vector<float> disp_ = theMuonModifier.gaussianRandomVector(dx, dy, dz);
    const std::vector<float> rotation_ = theMuonModifier.gaussianRandomVector(phix, phiy, phiz);
    disp.push_back(disp_[0]); disp.push_back(disp_[1]); disp.push_back(disp_[2]);
    rotation.push_back(rotation_[0]); rotation.push_back(rotation_[1]); rotation.push_back(rotation_[2]);
  } else if (dist_ == 1) {
    const std::vector<float> disp_ = theMuonModifier.flatRandomVector(dx, dy, dz);
    const std::vector<float> rotation_ = theMuonModifier.flatRandomVector(phix, phiy, phiz);
    disp.push_back(disp_[0]); disp.push_back(disp_[1]); disp.push_back(disp_[2]);
    rotation.push_back(rotation_[0]); rotation.push_back(rotation_[1]); rotation.push_back(rotation_[2]);
  } else {
    disp.push_back(dx); disp.push_back(dy); disp.push_back(dz);
    rotation.push_back(phix); rotation.push_back(phiy); rotation.push_back(phiz);
  }
  for(std::vector<Alignable *>::iterator iter = DTbarrel.begin(); iter != DTbarrel.end(); ++iter) {
    theMuonModifier.moveAlignable( *iter, false, true, disp[0], disp[1], disp[2] );
    theMuonModifier.rotateAlignable( *iter, false, true, rotation[0],  rotation[1], rotation[2] );
    theMuonModifier.addAlignmentPositionError( *iter, errorx, errory, errorz );
    theMuonModifier.addAlignmentPositionErrorFromRotation( *iter,  errorphix, errorphiy, errorphiz ); 
  }
  for(std::vector<Alignable *>::iterator iter = CSCendcaps.begin(); iter != CSCendcaps.end(); ++iter) {
    theMuonModifier.moveAlignable( *iter, false, true, disp[0], disp[1], disp[2] );
    theMuonModifier.rotateAlignable( *iter, false, true, rotation[0],  rotation[1], rotation[2] );
    theMuonModifier.addAlignmentPositionError( *iter, errorx, errory, errorz );
    theMuonModifier.addAlignmentPositionErrorFromRotation( *iter,  errorphix, errorphiy, errorphiz ); 
  }
}


//______________________________________________________________________________________________________
void MuonScenarioBuilder::moveChamberInSector(Alignable *chamber, align::Scalars disp, align::Scalars rotation, align::Scalars dispError, align::Scalars rotationError)
{
    align::RotationType rotx( Basic3DVector<double>(1.0, 0.0, 0.0), rotation[0] );
    align::RotationType roty( Basic3DVector<double>(0.0, 1.0, 0.0), rotation[1] );
    align::RotationType rotz( Basic3DVector<double>(0.0, 0.0, 1.0), rotation[2] );
    align::RotationType rot = rotz * roty * rotx;
    align::GlobalPoint pos = chamber->globalPosition();
    align::GlobalPoint dispRot(pos.basicVector()-rot*pos.basicVector());
    disp[0] += dispRot.x(); disp[1] += dispRot.y(); disp[2] += dispRot.z();
    theMuonModifier.moveAlignable( chamber, false, true, disp[0], disp[1], disp[2] );
    theMuonModifier.rotateAlignable( chamber, false, true, rotation[0],  rotation[1], rotation[2] );
    theMuonModifier.addAlignmentPositionError( chamber, dispError[0], dispError[1], dispError[2] );
    theMuonModifier.addAlignmentPositionErrorFromRotation( chamber,  rotationError[0], rotationError[1], rotationError[2] );
}

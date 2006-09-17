#include "CondFormats/SiPixelObjects/interface/FrameConversion.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;
using namespace sipixelobjects;

FrameConversion::FrameConversion( int rowOffset, int rowSlopeSign, int colOffset, int colSlopeSign)
  : theRowConversion( LinearConversion(rowOffset,rowSlopeSign) ),
    theCollumnConversion( LinearConversion(colOffset, colSlopeSign) )
{ }

FrameConversion::FrameConversion( const PixelBarrelName & name, int rocIdInDetUnit)
{
  int slopeRow =0;
  int slopeCol = 0;
  int  rowOffset = 0;
  int  colOffset = 0; 
  if (name.isHalfModule() ) {
    slopeRow = -1; 
    slopeCol = 1;
    rowOffset = PixelROC::rows()-1;
    colOffset = rocIdInDetUnit * PixelROC::cols(); 
  } else {
    if (rocIdInDetUnit <8) {
      slopeRow = -1;
      slopeCol = 1;
      rowOffset = 2*PixelROC::rows()-1;
      colOffset = rocIdInDetUnit * PixelROC::cols(); 
    } else {
      slopeRow = 1;
      slopeCol = -1;
      rowOffset = 0;
      colOffset = (16-rocIdInDetUnit)*PixelROC::cols()-1; 
    }
  } 

  //
  // FIX for negative barrel (not inverted modules)
  //
  PixelBarrelName::Shell shell = name.shell();
  if (shell == PixelBarrelName::mO || shell == PixelBarrelName::mI) {
    slopeRow *= -1;
    slopeCol *= -1;
    colOffset = 8*PixelROC::cols()-colOffset-1;
    switch(name.moduleType()) {
      case(PixelModuleName::v1x8) : { rowOffset =   PixelROC::rows()-rowOffset-1; break; }
      default:                      { rowOffset = 2*PixelROC::rows()-rowOffset-1; break; }
    }
  } 

  theRowConversion      = LinearConversion(rowOffset,slopeRow);
  theCollumnConversion =  LinearConversion(colOffset, slopeCol);

}
FrameConversion::FrameConversion( const PixelEndcapName & name, int rocIdInDetUnit)
{
  int slopeRow =0;
  int slopeCol = 0;
  int  rowOffset = 0;
  int  colOffset = 0; 

  if (name.pannelName()==1) {
    if (name.plaquetteName()==1) {
      slopeRow = 1;
      slopeCol = -1;
      rowOffset = 0;
      colOffset = (2-rocIdInDetUnit)*PixelROC::cols()-1;
    } else if (name.plaquetteName()==2) {
      if (rocIdInDetUnit <3) {
        slopeRow = -1;
        slopeCol = 1;
        rowOffset = 2*PixelROC::rows()-1;
        colOffset = rocIdInDetUnit*PixelROC::cols();
      } else {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (6-rocIdInDetUnit)*PixelROC::cols()-1;
      }
    } else if (name.plaquetteName()==3) {
      if (rocIdInDetUnit <4) {
        slopeRow = -1;
        slopeCol = 1;
        rowOffset = 2*PixelROC::rows()-1;
        colOffset = rocIdInDetUnit*PixelROC::cols();
      } else {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (8-rocIdInDetUnit)*PixelROC::cols()-1;
      }
    } else if (name.plaquetteName()==4) {
      slopeRow = -1;
      slopeCol = 1;
      rowOffset = PixelROC::rows()-1;
      colOffset = rocIdInDetUnit*PixelROC::cols();
    }
  } else {
    if (name.plaquetteName()==1) {
      if (rocIdInDetUnit <3) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (3-rocIdInDetUnit)*PixelROC::cols()-1;
      } else {
        slopeRow = -1;
        slopeCol = 1;
        colOffset = (rocIdInDetUnit-3)*PixelROC::cols();
        rowOffset = 2*PixelROC::rows()-1;
      } 
    } else if (name.plaquetteName()==2) {
      if (rocIdInDetUnit <4) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (4-rocIdInDetUnit)*PixelROC::cols()-1;
      } else {
        slopeRow = -1;
        slopeCol = 1;
        colOffset = (rocIdInDetUnit-4)*PixelROC::cols();
        rowOffset = 2*PixelROC::rows()-1;
      } 
    } else if (name.plaquetteName()==3) {
      if (rocIdInDetUnit <5) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (5-rocIdInDetUnit)*PixelROC::cols()-1;
      } else {
        slopeRow = -1;
        slopeCol = 1;
        colOffset = (rocIdInDetUnit-5)*PixelROC::cols();
        rowOffset = 2*PixelROC::rows()-1;
      } 
    }
  }

  theRowConversion =  LinearConversion(rowOffset,slopeRow);
  theCollumnConversion =  LinearConversion(colOffset, slopeCol);
}

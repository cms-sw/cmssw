#include "CondFormats/SiPixelObjects/interface/FrameConversion.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;
using namespace sipixelobjects;

FrameConversion::FrameConversion( const PixelBarrelName & name, int rocIdInDetUnit)
{
  int slopeRow =0;
  int slopeCol = 0;
  int  rowOffset = 0;
  int  colOffset = 0; 
  if (name.isHalfModule() ) {
    slopeRow = -1; 
    slopeCol = 1;
    rowOffset = LocalPixel::numRowsInRoc-1;
    colOffset = rocIdInDetUnit * LocalPixel::numColsInRoc; 
  } else {
    if (rocIdInDetUnit <8) {
      slopeRow = -1;
      slopeCol = 1;
      rowOffset = 2*LocalPixel::numRowsInRoc-1;
      colOffset = rocIdInDetUnit * LocalPixel::numColsInRoc; 
    } else {
      slopeRow = 1;
      slopeCol = -1;
      rowOffset = 0;
      colOffset = (16-rocIdInDetUnit)*LocalPixel::numColsInRoc-1; 
    }
  } 

  //
  // FIX for negative barrel (not inverted modules)
  //
  PixelBarrelName::Shell shell = name.shell();
  if (shell == PixelBarrelName::mO || shell == PixelBarrelName::mI) {
    slopeRow *= -1;
    slopeCol *= -1;
    colOffset = 8*LocalPixel::numColsInRoc-colOffset-1;
    switch(name.moduleType()) {
      //case(PixelModuleName::v1x8) : { rowOffset =   LocalPixel::numRowsInRoc-rowOffset-1; break; }
      case(PixelModuleName::v1x8) : { 
	slopeRow = -1;  // d.k. 23/10/08 
	slopeCol = 1;   // d.k. 13/11/08
	//colOffset = rocIdInDetUnit * LocalPixel::numColsInRoc;  // d.k. 13/11/08
	//colOffset = (8-rocIdInDetUnit) * LocalPixel::numColsInRoc -1;  // d.k. 19/11/08
	colOffset = (8-rocIdInDetUnit-1) * LocalPixel::numColsInRoc;  // d.k. 19/11/08
	//cout<<" FramConversion: "<<rocIdInDetUnit<<" "<<slopeRow<<" "<<slopeCol<<" "<<rowOffset<<" "<<colOffset<<endl;
	break; 
      } 
      default:                      { rowOffset = 2*LocalPixel::numRowsInRoc-rowOffset-1; break; }
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
      colOffset = (1+rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
    } else if (name.plaquetteName()==2) {
      if (rocIdInDetUnit <3) {
        slopeRow = -1;
        slopeCol = 1;
        rowOffset = 2*LocalPixel::numRowsInRoc-1;
        colOffset = rocIdInDetUnit*LocalPixel::numColsInRoc;
      } else {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (6-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
      }
    } else if (name.plaquetteName()==3) {
      if (rocIdInDetUnit <4) {
        slopeRow = -1;
        slopeCol = 1;
        rowOffset = 2*LocalPixel::numRowsInRoc-1;
        colOffset = rocIdInDetUnit*LocalPixel::numColsInRoc;
      } else {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (8-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
      }
    } else if (name.plaquetteName()==4) {
      slopeRow = -1;
      slopeCol = 1;
      rowOffset = LocalPixel::numRowsInRoc-1;
      colOffset = rocIdInDetUnit*LocalPixel::numColsInRoc;
    }
  } else {
    if (name.plaquetteName()==1) {
      if (rocIdInDetUnit <3) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (3-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
      } else {
        slopeRow = -1;
        slopeCol = 1;
        colOffset = (rocIdInDetUnit-3)*LocalPixel::numColsInRoc;
        rowOffset = 2*LocalPixel::numRowsInRoc-1;
      } 
    } else if (name.plaquetteName()==2) {
      if (rocIdInDetUnit <4) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (4-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
      } else {
        slopeRow = -1;
        slopeCol = 1;
        colOffset = (rocIdInDetUnit-4)*LocalPixel::numColsInRoc;
        rowOffset = 2*LocalPixel::numRowsInRoc-1;
      } 
    } else if (name.plaquetteName()==3) {
      if (rocIdInDetUnit <5) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (5-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
      } else {
        slopeRow = -1;
        slopeCol = 1;
        colOffset = (rocIdInDetUnit-5)*LocalPixel::numColsInRoc;
        rowOffset = 2*LocalPixel::numRowsInRoc-1;
      } 
    }
  }

  theRowConversion =  LinearConversion(rowOffset,slopeRow);
  theCollumnConversion =  LinearConversion(colOffset, slopeCol);
}

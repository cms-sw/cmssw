#include "CondFormats/SiPixelObjects/interface/FrameConversion.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;
using namespace sipixelobjects;

FrameConversion::FrameConversion(bool bpix, int side, int layer, int rocIdInDetUnit) {
  int slopeRow =0;
  int slopeCol = 0;
  int  rowOffset = 0;
  int  colOffset = 0; 
 
  if (bpix ) { // bpix 
    
    if (side==-1 && layer!=1) {  // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1

      if (rocIdInDetUnit <8) {
	slopeRow = 1;
	slopeCol = -1;
	rowOffset = 0;
	colOffset = (8-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;	
      } else {
	slopeRow = -1;
	slopeCol = 1;      
	rowOffset = 2*LocalPixel::numRowsInRoc-1;
	colOffset = (rocIdInDetUnit-8)*LocalPixel::numColsInRoc;
      } // if roc
      
    } else {  // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in Layer 1
      
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
      
    } // end if +-Z



  } else { // fpix 

    // for fpix follow Urs's code for pilot blade
    // no difference between panels
    if(side==-1) { // pannel 1
      if (rocIdInDetUnit < 8) {
	slopeRow = 1;
	slopeCol = -1;
	rowOffset = 0;
	colOffset = (8-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
      } else {
	slopeRow = -1;
	slopeCol = 1;
	rowOffset = 2*LocalPixel::numRowsInRoc-1;
	colOffset = (rocIdInDetUnit-8)*LocalPixel::numColsInRoc;
      }
    } else { // pannel 2 
      if (rocIdInDetUnit < 8) {
	slopeRow = 1;
	slopeCol = -1;
	rowOffset = 0;
	colOffset = (8-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
      } else {
	slopeRow = -1;
	slopeCol = 1;
	rowOffset = 2*LocalPixel::numRowsInRoc-1;
	colOffset = (rocIdInDetUnit-8)*LocalPixel::numColsInRoc;
      }
      
    } // side 

  } // bpix/fpix

  theRowConversion      = LinearConversion(rowOffset,slopeRow);
  theCollumnConversion =  LinearConversion(colOffset, slopeCol);
  
}

FrameConversion::FrameConversion(bool bpix, int side, int rocIdInDetUnit) {
  int slopeRow =0;
  int slopeCol = 0;
  int  rowOffset = 0;
  int  colOffset = 0; 
 
  if (bpix ) { // bpix 
    
    if (side==-1) {  // -Z side

      if (rocIdInDetUnit <8) {
	slopeRow = 1;
	slopeCol = -1;
	rowOffset = 0;
	colOffset = (8-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;	
      } else {
	slopeRow = -1;
	slopeCol = 1;      
	rowOffset = 2*LocalPixel::numRowsInRoc-1;
	colOffset = (rocIdInDetUnit-8)*LocalPixel::numColsInRoc;
      } // if roc
      
    } else {  // +Z side
      
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
      
    } // end if +-Z



  } else { // fpix 

    // for fpix follow Urs's code for pilot blade
    // no difference between panels
    if(side==-1) { // pannel 1
      if (rocIdInDetUnit < 8) {
	slopeRow = 1;
	slopeCol = -1;
	rowOffset = 0;
	colOffset = (8-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
      } else {
	slopeRow = -1;
	slopeCol = 1;
	rowOffset = 2*LocalPixel::numRowsInRoc-1;
	colOffset = (rocIdInDetUnit-8)*LocalPixel::numColsInRoc;
      }
    } else { // pannel 2 
      if (rocIdInDetUnit < 8) {
	slopeRow = 1;
	slopeCol = -1;
	rowOffset = 0;
	colOffset = (8-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;
      } else {
	slopeRow = -1;
	slopeCol = 1;
	rowOffset = 2*LocalPixel::numRowsInRoc-1;
	colOffset = (rocIdInDetUnit-8)*LocalPixel::numColsInRoc;
      }
      
    } // side 

  } // bpix/fpix

  theRowConversion      = LinearConversion(rowOffset,slopeRow);
  theCollumnConversion =  LinearConversion(colOffset, slopeCol);
  
}

// OLD method for phase0 bpix
FrameConversion::FrameConversion( const PixelBarrelName & name, int rocIdInDetUnit)
{
  int slopeRow =0;
  int slopeCol = 0;
  int  rowOffset = 0;
  int  colOffset = 0; 

  //
  PixelBarrelName::Shell shell = name.shell();
  if (shell == PixelBarrelName::mO || shell == PixelBarrelName::mI) {  // -Z side

    if (name.isHalfModule() ) {

      slopeRow = -1;  // d.k. 23/10/08 
      slopeCol = 1;   // d.k. 13/11/08
      rowOffset = LocalPixel::numRowsInRoc-1;
      colOffset = rocIdInDetUnit * LocalPixel::numColsInRoc;  // d.k. 13/11/08

    } else {

      if (rocIdInDetUnit <8) {
	slopeRow = 1;
	slopeCol = -1;

	rowOffset = 0;
	colOffset = (8-rocIdInDetUnit)*LocalPixel::numColsInRoc-1;

      } else {
	slopeRow = -1;
	slopeCol = 1;

	rowOffset = 2*LocalPixel::numRowsInRoc-1;
	colOffset = (rocIdInDetUnit-8)*LocalPixel::numColsInRoc;

      }
    } 
    

  } else {  // +Z side

    if (name.isHalfModule() ) {
      slopeRow = -1; 
      slopeCol = 1;
      rowOffset = LocalPixel::numRowsInRoc-1;
      colOffset = rocIdInDetUnit * LocalPixel::numColsInRoc; 
    } else {  // Full modules
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
    } // if modules 
    

  } // end if +-Z

  theRowConversion      = LinearConversion(rowOffset,slopeRow);
  theCollumnConversion =  LinearConversion(colOffset, slopeCol);

}
// OLD method for phase0 fpix
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

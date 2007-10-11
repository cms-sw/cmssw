#ifndef Alignment_CommonAlignment_AlignablesFromGeometry_H
#define Alignment_CommonAlignment_AlignablesFromGeometry_H

/** \class AlignablesFromGeometry
 *
 *  A module to build alignable units (lowest components).
 *
 *  Usage:
 *    module sensors = AlignablesFromGeometry
 *    {
 *      untracked PSet tracker =
 *      {
 *        string pixelBarrel = "TPBsensors"
 *        string pixelEndcap = "TPEsensors"
 *        string innerBarrel = "TIBmodules"
 *        string innerEndcap = "TIDmodules"
 *        string outerBarrel = "TOBmodules"
 *        string outerEndcap = "TECmodules"
 *      }
 *
 *      untracked PSet muonDet =
 *      {
 *        string muonBarrel =  "DTlayers"
 *        string muonEndcap = "CSClayers"
 *      }
 *    }
 *
 *  This module just allows user to specify the names for the list of basic
 *  components in each sub-detector so that these lists can be used in other
 *  modules. For eg, in AlignableBuilder.
 *
 *  Names must be unique.
 *
 *  $Date: 2007/10/08 16:21:12 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

class AlignablesFromGeometry:
  public edm::EDAnalyzer
{
  typedef TrackingGeometry::DetContainer DetContainer;

  public:

  /// Init cfg parameters.
  AlignablesFromGeometry(
			 const edm::ParameterSet&
			 );

  /// Build the Alignables.
  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       );

  private:

  /// Convert GeomDets to Alignables.
  static void detsToAlignables(
			       const DetContainer&,
			       const std::string& name
			       );

  /// Build a map from id to Alignable* that are built from GeomDets.
  static void buildAlignableMap(
				const align::Alignables&
				);

  edm::ParameterSet theTrackerCfg;
  edm::ParameterSet theMuonDetCfg;
};

#endif

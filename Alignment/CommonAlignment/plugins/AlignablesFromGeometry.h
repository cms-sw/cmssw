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
 *  modules. For eg, in AlignableCompositeBuilder.
 *
 *  Names must be unique.
 *
 *  $Date: 2007/04/25 18:37:59 $
 *  $Revision: 1.8 $
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

  /// Set file name
  AlignablesFromGeometry(
			 const edm::ParameterSet&
			 );

  /// Read from DB and print survey info.
  virtual void beginJob(
			const edm::EventSetup&
			);

  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       ) {}

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

#ifndef Alignment_CommonAlignment_TrackerFromGeometry_H
#define Alignment_CommonAlignment_TrackerFromGeometry_H

/** \class TrackerFromGeometry
 *
 *  Module to build the tracker from geometry.
 *
 *  Usage:
 *    module tracker = TrackerFromGeometry
 *    {
 *      untracked bool applyAlignment = true
 *    }
 *
 *  Set applyAlignment to true to apply alignments from DB. Default is false.
 *
 *  $Date: 2007/10/08 16:21:12 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

class TrackerFromGeometry:
  public edm::EDAnalyzer
{
  public:

  /// Set the flag applyAlignment.
  TrackerFromGeometry(
		      const edm::ParameterSet&
		      );

  /// Create the tracker geometry.
  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       );

  private:

  bool applyAlignment_; // true to apply alignments from DB
};

#endif

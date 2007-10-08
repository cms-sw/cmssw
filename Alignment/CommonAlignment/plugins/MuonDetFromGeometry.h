#ifndef Alignment_CommonAlignment_MuonDetFromGeometry_H
#define Alignment_CommonAlignment_MuonDetFromGeometry_H

/** \class MuonDetFromGeometry
 *
 *  Module to build the muon detector from geometry.
 *
 *  Usage:
 *    module muon = MuonDetFromGeometry
 *    {
 *      untracked bool applyAlignment = true
 *    }
 *
 *  Set applyAlignment to true to apply alignments from DB. Default is false.
 *
 *  $Date: 2007/04/25 18:37:59 $
 *  $Revision: 1.8 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

class MuonDetFromGeometry:
  public edm::EDAnalyzer
{
  public:

  /// Set the flag applyAlignment.
  MuonDetFromGeometry(
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

  bool applyAlignment_; // true to apply alignments from DB
};

#endif

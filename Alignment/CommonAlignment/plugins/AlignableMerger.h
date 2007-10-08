#ifndef Alignment_CommonAlignment_AlignableMerger_H
#define Alignment_CommonAlignment_AlignableMerger_H

/** \class AlignableMerger
 *
 *  A module to merge lists of alignables into one list.
 *
 *  Usage:
 *    module tracker = AlignableMerger
 *    {
 *      vstring inLists = {"pixel", "strip"}
 *    }
 *
 *  A new list called "tracker" is created which contains all the alignables
 *  in the lists "pixel" and "strip".
 *
 *  $Date: 2007/04/25 18:37:59 $
 *  $Revision: 1.8 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

class AlignableMerger:
  public edm::EDAnalyzer
{
  typedef std::vector<std::string> Strings;

  public:

  AlignableMerger(
		  const edm::ParameterSet&
		  );

  virtual void beginJob(
			const edm::EventSetup&
			) {}

  virtual void analyze(
		       const edm::Event&,
		       const edm::EventSetup&
		       );

  private:

  Strings theInLists;

  std::string theOutList;
};

#endif

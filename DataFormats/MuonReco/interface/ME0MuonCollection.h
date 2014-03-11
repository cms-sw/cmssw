#ifndef DataFormats_ME0MuonCollection_H
#define DataFormats_ME0MuonCollection_H

/** \class ME0MuonCollection
 *
 * The collection of ME0Muon's. See \ref ME0MuonCollection.h for details.
 *
 *  $Date: 2010/03/12 13:08:15 $
 *  \author Matteo Sani
 */

#include "DataFormats/MuonReco/interface/ME0Muon.h"
#include "DataFormats/Common/interface/Ref.h"

/// collection of ME0Muons
typedef std::vector<reco::ME0Muon> ME0MuonCollection;

/// persistent reference to a ME0Muon
typedef edm::Ref<ME0MuonCollection> ME0MuonRef;
	
#endif

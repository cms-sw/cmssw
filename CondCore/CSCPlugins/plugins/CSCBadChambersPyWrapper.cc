#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <sstream>

namespace cond {
//Get CSCDetId from these functions:
//
//  CSCDetId detIdFromLayerIndex( IndexType ili ) const;
//  CSCDetId detIdFromChamberIndex( IndexType ici ) const;
//  CSCDetId detIdFromChamberIndex_OLD( IndexType ici ) const;
//  CSCDetId detIdFromChamberLabel( IndexType ie, IndexType icl ) const;
//  std::pair<CSCDetId, IndexType> detIdFromStripChannelIndex( LongIndexType ichi ) const;
//  std::pair<CSCDetId, IndexType> detIdFromChipIndex( IndexType ichi ) const;

	template<>
	std::string PayLoadInspector<CSCBadChambers>::summary() const {
		std::stringstream ss;
		CSCIndexer indexer;
		/// How many bad chambers are there
		int numberOfChambers = object().numberOfChambers();

		/// Return the container of bad chambers
		std::vector<int> vContainerInts = object().container(); 
		std::vector<CSCDetId> vCSCDetIds;
		std::vector<int>::const_iterator iInt;
		//get vector of CSCDetId:
		for (iInt = vContainerInts.begin(); iInt != vContainerInts.end(); ++iInt){
			vCSCDetIds.push_back(indexer.detIdFromChamberIndex(*iInt));
		}
		/// Is the given chamber flagged as bad?
		//bool isInBadChamber0 = object().isInBadChamber( const CSCDetId& id );

		//print data:
		ss << "---Total of bad Chambers: " << numberOfChambers << std::endl;
		ss << "--Bad chambers:" << std::endl;
		iInt = vContainerInts.begin();
		int index = 0;
		for (std::vector<CSCDetId>::const_iterator iCSCDetId = vCSCDetIds.begin(); iCSCDetId != vCSCDetIds.end(); ++iCSCDetId, ++iInt){
			ss << index++ << ". "<< "Id[" << *iInt << 
				"]-> Chamber index: " << indexer.chamberIndex( *iCSCDetId ) <<
				"; Layer index: " << indexer.layerIndex( *iCSCDetId ) << ";"<< std::endl;

		}

		return ss.str();
	}

}
PYTHON_WRAPPER(CSCBadChambers,CSCBadChambers);
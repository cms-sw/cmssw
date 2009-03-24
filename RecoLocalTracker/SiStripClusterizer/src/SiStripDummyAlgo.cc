#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripDummyAlgo.h"
#include <iostream>
#include <sstream>
#include <vector>

// -----------------------------------------------------------------------------
//
SiStripDummyAlgo::SiStripDummyAlgo( const edm::ParameterSet& pset ) 
  : SiStripClusterizerAlgo(pset) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripDummyAlgo::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
SiStripDummyAlgo::~SiStripDummyAlgo() {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripDummyAlgo::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void SiStripDummyAlgo::clusterize( const DigisDSnew& digis, 
				   ClustersV& clusters ) {
  
  // Some initialization
  clusters.clear();
  
  if ( digis.empty() ) { return; }

  DigisV::const_iterator begin( digis.begin() );
  DigisV::const_iterator end( digis.end() );
  SiStripCluster::SiStripDigiRange range( begin, begin );
  
//   // Iterates through digis and builds clusters from non-zero digis
//       std::cout << " GET HERE1! " << std::endl;
//   DigisV::const_iterator idigi( begin );
//   DigisV::const_iterator jdigi( end );
//   for ( ; idigi != jdigi; ++idigi ) {
//     if ( idigi->adc() ) { 
//       std::cout << " GET HERE3! " << std::endl;
//       if ( range.first == range.second ) { 
// 	std::cout << " GET HERE4! " << std::endl;
// 	range.first = idigi; 
//       }
//       range.second = idigi+1;
//     } else {
//       std::cout << " GET HERE5! " << std::endl;
//       if ( range.first != range.second ) { 
// 	std::cout << " GET HERE6! " << std::endl;
// 	clusters.push_back( SiStripCluster( digis.id(), range ) ); 
//       }
//       range.first = end;
//       range.second = end;
//     }
//   }

//   // Iterates through digis and builds clusters from non-zero digis
//   DigisV::const_iterator idigi( begin );
//   DigisV::const_iterator jdigi( end );
//   for ( ; idigi != jdigi; ++idigi ) {
//     if ( idigi == begin ) { // first digi
//       range.first = idigi; 
//       range.second = idigi+1; 
//     } else if ( idigi+1 == end ) { // last digi
//       range.second++;
//       clusters.push_back( SiStripCluster( digis.id(), range ) ); 
//     } else { // all other digis
//       if ( idigi->strip() - (idigi-1)->strip() == 1 ) { range.second++; } 
//       else { 
// 	clusters.push_back( SiStripCluster( digis.id(), range ) ); 
// 	range.first = idigi; 
// 	range.second = idigi+1; 
//       }
//     }
//   }

//   // Iterates through digis and builds clusters from non-zero digis
//   DigisV::const_iterator idigi( begin );
//   DigisV::const_iterator jdigi( end );
//   for ( ; idigi != jdigi; ++idigi ) {
//     if ( idigi != begin ) {
//       if ( idigi != end ) {
// 	if ( idigi->strip() - (idigi-1)->strip() > 1 ) { // check for gap
// 	  clusters.push_back( SiStripCluster( digis.id(), range ) ); 
// 	  range.first = idigi;
// 	}
// 	range.second = idigi+1;
//     } else { range.second = idigi+1; }
//   }
//   if ( range.first != range.second) { clusters.push_back( SiStripCluster( digis.id(), range ) ); }

  // Iterates through digis and builds clusters from non-zero digis
  DigisV::const_iterator idigi( begin );
  DigisV::const_iterator jdigi( end );
  for ( ; idigi != jdigi; ++idigi ) {
    if ( idigi != begin ) {
      if ( idigi->strip() - (idigi-1)->strip() > 1 ) { // check for gap
	clusters.push_back( SiStripCluster( digis.id(), range ) ); 
	range.first = idigi; 
      } 
    }
    range.second = idigi+1;
  }
  if ( range.first != range.second ) { clusters.push_back( SiStripCluster( digis.id(), range ) ); }
  
//   DigisV test;
    
//   std::stringstream ss;
//   ss << " TEST id: " << digis.id() << " size: " << clusters.size();
//   bool null = false;
//   ClustersV::const_iterator ii = clusters.begin();
//   ClustersV::const_iterator jj = clusters.end();
//   for ( ; ii != jj; ++ii ) {
//     ss << " #: " << uint16_t( ii - clusters.begin() ) 
//        << " size: " << ii->amplitudes().size()
//        << " str: " << ii->firstStrip() << " amp: ";
//     std::vector<uint8_t>::const_iterator iii = ii->amplitudes().begin();
//     std::vector<uint8_t>::const_iterator jjj = ii->amplitudes().end();
//     for ( ; iii != jjj; ++iii ) {
//       uint16_t strip = ii->firstStrip() + uint16_t( iii - ii->amplitudes().begin() );
//       test.push_back( SiStripDigi( strip, *iii ) );
//       ss << " " << uint16_t(*iii);
//       if ( !(*iii) ) { null = true; }
//     }
//   }

//   {
//     std::stringstream ss;
//     ss << " TESTTEST " 
//        << std::endl;
//     if ( end - begin != test.end() - test.begin() ) {
//       ss << " clusters.size(): " << clusters.size()
// 	 << " digis.size(): " << uint16_t( end - begin )
// 	 << " test.size(): " << test.size()
// 	 << std::endl;
//       ss << " ORIG id: " << digis.id();
//       DigisV::const_iterator idigi( begin );
//       DigisV::const_iterator jdigi( end );
//       for ( ; idigi != jdigi; ++idigi ) {
// 	ss << " " << idigi->strip() << "/" << idigi->adc();
//       }
//       ss << std::endl;
    
//       ss << " NEW  id: " << digis.id();
//       DigisV::const_iterator id( test.begin() );
//       DigisV::const_iterator jd( test.end() );
//       for ( ; id != jd; ++id ) {
// 	ss << " " << id->strip() << "/" << id->adc();
//       }
//     } else {
//       ss << " SAME! ";
//     }
//     ss << std::endl;
//     std::cout << ss.str() << std::endl;
//   }

//   if ( null ) {
//     edm::LogWarning("UNDEFINED_CATEGORY")
//       << "[SiStripDummyAlgo::" << __func__ << "]"
//       << " NULL value for id " << digis.id() << "!";
//   }
  
//   if ( clusters.size() ) {
//     edm::LogWarning("UNDEFINED_CATEGORY")
//       << "[SiStripDummyAlgo::" << __func__ << "]"
//       << std::endl << ss.str() << std::endl;
//   }
    
}

// -----------------------------------------------------------------------------
//
void SiStripDummyAlgo::add( ClustersV& clusters,
			    const uint32_t& id, 
			    const uint16_t& strip,
			    const uint16_t& adc ) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripDummyAlgo::" << __func__ << "]";
}


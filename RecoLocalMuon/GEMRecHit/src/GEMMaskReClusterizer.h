#ifndef RecoLocalMuon_GEMMaskReClusterizer_h
#define RecoLocalMuon_GEMMaskReClusterizer_h

/** \Class GEMMaskReClusterizer
 *  $Date: 2013/04/04 13:35:55 $
 *  $Revision: 1.2 $
 *  \author J.C. Sanabria -- UniAndes, Bogota
 */

#include "GEMEtaPartitionMask.h"

#include "GEMCluster.h"
#include "GEMClusterizer.h"
#include "GEMClusterContainer.h"

#include "DataFormats/MuonDetId/interface/GEMDetId.h"


class GEMMaskReClusterizer 
{
 public :

   GEMMaskReClusterizer();
   ~GEMMaskReClusterizer();
   GEMClusterContainer doAction(const GEMDetId& ,GEMClusterContainer& , const EtaPartitionMask& );
   int get(const EtaPartitionMask& ,int );

};

#endif

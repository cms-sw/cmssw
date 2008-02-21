#ifndef Alignment_CommonAlignment_MuonNameSpace_H
#define Alignment_CommonAlignment_MuonNameSpace_H

/** \namespace muon
 *
 *  Namespace for numbering sub-detectors in Muon.
 *
 *  Numbering starts from 1.
 *
 *  $Date: 2007/10/18 09:57:10 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

namespace align
{
  namespace muon
  {
    enum
    {
      DT  = MuonSubdetId::DT,
      CSC = MuonSubdetId::CSC,
      RPC = MuonSubdetId::RPC,
			MAX = RPC + 1
    };
	}
}

#endif

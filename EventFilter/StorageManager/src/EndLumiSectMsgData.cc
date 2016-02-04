// $Id: EndLumiSectMsgData.cc,v 1.9 2011/03/07 15:31:32 mommsen Exp $
/// @file: EndLumiSectMsgData.cc

#include "EventFilter/StorageManager/src/ChainData.h"

#include "interface/evb/version.h"
#include "interface/evb/i2oEVBMsgs.h"
#include "interface/shared/i2oXFunctionCodes.h"
#include "interface/shared/version.h"


namespace stor
{

  namespace detail
  {

    EndLumiSectMsgData::EndLumiSectMsgData(toolbox::mem::Reference* pRef):
      ChainData(I2O_EVM_LUMISECTION),
      runNumber_(0),
      lumiSection_(0)
    {
      expectedNumberOfFragments_ = 1; //EoLS has always only one fragment
      addFirstFragment(pRef);
      // EoLS message from EVM does not have an initial header.
      // Thus, reset this faulty bit
      faultyBits_ &= ~CORRUPT_INITIAL_HEADER;

      if ( !faulty() )
      {
        I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME* msg_frame =
          (I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME*)( pRef->getDataLocation() );
        if (msg_frame)
        {
          runNumber_ = msg_frame->runNumber;
          lumiSection_ = msg_frame->lumiSection;
        }
      }
    }

  } // namespace detail

} // namespace stor



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

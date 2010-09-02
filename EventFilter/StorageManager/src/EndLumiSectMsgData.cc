// $Id: EndLumiSectMsgData.cc,v 1.6 2010/05/17 15:59:10 mommsen Exp $
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
      _runNumber(0),
      _lumiSection(0)
    {
      _expectedNumberOfFragments = 1; //EoLS has always only one fragment
      addFirstFragment(pRef);
      // EoLS message from EVM does not have an initial header.
      // Thus, reset this faulty bit
      _faultyBits &= ~CORRUPT_INITIAL_HEADER;

      if ( !faulty() )
      {
        I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME* msg_frame =
          (I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME*)( pRef->getDataLocation() );
        if (msg_frame)
        {
          _runNumber = msg_frame->runNumber;
          _lumiSection = msg_frame->lumiSection;
        }
      }
    }

    uint32_t EndLumiSectMsgData::do_runNumber() const
    {
      return _runNumber;
    }

    uint32_t EndLumiSectMsgData::do_lumiSection() const
    {
      return _lumiSection;
    }

  } // namespace detail

} // namespace stor



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

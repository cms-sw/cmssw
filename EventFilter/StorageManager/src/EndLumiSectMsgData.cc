// $Id: $

#include "EventFilter/StorageManager/src/ChainData.h"

namespace stor
{

  namespace detail
  {

    typedef struct _I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME
    {
      I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
      U32 runNumber;
      U32 lumiSection;
    }
      I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME,
      *PI2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME;

    EndLumiSectMsgData::EndLumiSectMsgData( toolbox::mem::Reference* pRef ):
      ChainData( pRef )
    {
      I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME* msg_frame
	= (I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME*)( pRef->getDataLocation() );
      _runNumber = msg_frame->runNumber;
      _lumiSection = msg_frame->lumiSection;
    }

    uint32 EndLumiSectMsgData::do_runNumber() const
    {
      return _runNumber;
    }

    uint32 EndLumiSectMsgData::do_lumiSection() const
    {
      return _lumiSection;
    }

  } // namespace detail

} // namespace stor

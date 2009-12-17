// $Id: EndLumiSectMsgData.cc,v 1.1 2009/10/12 13:15:29 dshpakov Exp $

#include "EventFilter/StorageManager/src/ChainData.h"

#include "interface/evb/version.h"
#include "interface/evb/i2oEVBMsgs.h"


namespace stor
{

  namespace detail
  {

    EndLumiSectMsgData::EndLumiSectMsgData( toolbox::mem::Reference* pRef ):
      ChainData( pRef ),
      _runNumber(0),
      _lumiSection(0)
    {
      #if (INTERFACEEVB_VERSION_MAJOR*1000 + INTERFACEEVB_VERSION_MINOR)>1008

      I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME* msg_frame
	= (I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME*)( pRef->getDataLocation() );
      _runNumber = msg_frame->runNumber;
      _lumiSection = msg_frame->lumiSection;

      #endif
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



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

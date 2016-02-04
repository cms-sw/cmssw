#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"

ESKCHIPBlock::ESKCHIPBlock()
{
  kId_ = -1;
  dccId_ = -1;
  fedId_ = -1;
  fiberId_ = -1;
  BC_ = -1;
  EC_ = -1;
  OptoBC_ = -1;
  OptoEC_ = -1;
  flag1_ = -1;
  flag2_ = -1;
  CRC_ = -1;

}

ESKCHIPBlock::ESKCHIPBlock(const int& kId)
{
  kId_ = kId;
  dccId_ = -1;
  fedId_ = -1;
  fiberId_ = -1;
  BC_ = -1;
  EC_ = -1;
  OptoBC_ = -1;
  OptoEC_ = -1;
  flag1_ = -1;
  flag2_ = -1;
  CRC_ = -1;
}

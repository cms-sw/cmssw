#include "DQMServices/Core/interface/DQMOldReceiver.h"
#include "DQMServices/Core/src/DQMError.h"

DQMOldReceiver::DQMOldReceiver(const std::string &, int, const std::string &, int, bool)
  : store_ (0)
{}

DQMOldReceiver::DQMOldReceiver(void)
  : store_ (0)
{}

DQMOldReceiver::~DQMOldReceiver(void)
{}

bool
DQMOldReceiver::update(void)
{
  raiseDQMError("DQMOldReceiver", "DQMOldReceiver::update() is obsolete");
  return true;
}

bool
DQMOldReceiver::doMonitoring(void)
{
  raiseDQMError("DQMOldReceiver", "DQMOldReceiver::doMonitoring() is obsolete");
  return true;
}

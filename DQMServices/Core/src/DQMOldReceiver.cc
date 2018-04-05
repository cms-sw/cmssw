#include "DQMServices/Core/interface/DQMOldReceiver.h"
#include "DQMServices/Core/src/DQMError.h"

DQMOldReceiver::DQMOldReceiver(const std::string &, int, const std::string &, int, bool)
  : store_ (nullptr)
{}

DQMOldReceiver::DQMOldReceiver()
  : store_ (nullptr)
{}

DQMOldReceiver::~DQMOldReceiver() = default;

bool
DQMOldReceiver::update()
{
  raiseDQMError("DQMOldReceiver", "DQMOldReceiver::update() is obsolete");
  return true;
}

bool
DQMOldReceiver::doMonitoring()
{
  raiseDQMError("DQMOldReceiver", "DQMOldReceiver::doMonitoring() is obsolete");
  return true;
}

#ifndef CONDFORMATS_HLTOBJECTS_ALCARECOTRIGGERBITS_H
#define CONDFORMATS_HLTOBJECTS_ALCARECOTRIGGERBITS_H
#include <string>
#include <map>
class AlCaRecoTriggerBits{
public:
  AlCaRecoTriggerBits();
  ~AlCaRecoTriggerBits();
private:
  std::map<std::string,std::string> m_alcarecoTotrig;
};
#endif

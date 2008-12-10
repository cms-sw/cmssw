#ifndef CONDFORMATS_HLTOBJECTS_ALCARECOTRIGGERBITS_H
#define CONDFORMATS_HLTOBJECTS_ALCARECOTRIGGERBITS_H
#include <string>
#include <map>
#include <vector>

class AlCaRecoTriggerBits {
public:
  AlCaRecoTriggerBits();
  ~AlCaRecoTriggerBits();

  /// Compose several paths into one string :
  std::string compose(const std::vector<std::string> &paths) const;
  /// Decompose one value of map from concatenated string
  std::vector<std::string> decompose(const std::string &concatPaths) const;
  /// Delimeter for composing paths to one string in DB:
  static const std::string::value_type delimeter_;

  std::map<std::string,std::string> m_alcarecoToTrig;
};
#endif

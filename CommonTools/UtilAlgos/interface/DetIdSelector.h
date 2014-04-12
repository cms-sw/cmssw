#ifndef DetIdSelector_h
#define DetIdSelector_h

#include <string>
#include <vector>

class DetId;
namespace edm {
  class ParameterSet;
}

class DetIdSelector {
 public:
  DetIdSelector();
  DetIdSelector(const std::string& selstring);
  DetIdSelector(const std::vector<std::string>& selstrings);
  DetIdSelector(const edm::ParameterSet& selconfig);

  bool isSelected(const DetId& detid ) const;
  bool isSelected(const unsigned int& rawid) const;
  bool operator()(const DetId& detid ) const;
  bool operator()(const unsigned int& rawid) const;
  inline bool isValid() const { return m_selections.size()!=0;}

 private:

  void addSelection(const std::string& selstring);
  void addSelection(const std::vector<std::string>& selstrings);

  std::vector<unsigned int> m_selections;
  std::vector<unsigned int> m_masks;

};

#endif // DetIdSelector_h

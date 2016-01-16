#ifndef DetIdSelector_h
#define DetIdSelector_h

#include <string>
#include <vector>

class DetId;
class TrackerTopology;
namespace edm {
  class ParameterSet;
}

class DetIdSelector {
 public:
  DetIdSelector();
  DetIdSelector(const std::string& selstring);
  DetIdSelector(const std::string& selstring, const std::string label);
  DetIdSelector(const std::vector<std::string>& selstrings);
  DetIdSelector(const std::vector<std::string>& selstrings, const std::string label);
  DetIdSelector(const edm::ParameterSet& selconfig);

  bool isSelected(const DetId& detid ) const;
  bool isSelected(const unsigned int& rawid) const;
  bool isSelectedByWords(const DetId& detid, const TrackerTopology* tTopo ) const;
  bool isSelectedByWords(const unsigned int& rawid, const TrackerTopology* tTopo) const;
  bool operator()(const DetId& detid ) const;
  bool operator()(const unsigned int& rawid) const;
  inline bool isValid() const { return m_selections.size()!=0;}

 private:

  void addSelection(const std::string& selstring);
  void addSelection(const std::vector<std::string>& selstrings);
  void addSelectionByWords(const std::string& selstring);
  void addSelectionByWords(const std::vector<std::string>& selstrings);

  bool isSelectedByWordsPXB(std::string label, const DetId& detid, const TrackerTopology* tTopo) const ;
  bool isSelectedByWordsPXF(std::string label, const DetId& detid, const TrackerTopology* tTopo) const ;
  bool isSelectedByWordsTIB(std::string label, const DetId& detid, const TrackerTopology* tTopo) const ;
  bool isSelectedByWordsTOB(std::string label, const DetId& detid, const TrackerTopology* tTopo) const ;
  bool isSelectedByWordsTID(std::string label, const DetId& detid, const TrackerTopology* tTopo) const ;
  bool isSelectedByWordsTEC(std::string label, const DetId& detid, const TrackerTopology* tTopo) const ;
  bool isSame(std::string label, std::string selection, unsigned int comparison, unsigned int spaces) const;
  bool isInRange(std::string range, unsigned int comparison, unsigned int spaces) const;

  std::vector<unsigned int> m_selections;
  std::vector<unsigned int> m_masks;
  std::vector<std::string> m_labels;

};

#endif // DetIdSelector_h

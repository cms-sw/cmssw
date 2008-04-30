#ifndef DataFormats_BTauReco_HLTOpenBJet_h
#define DataFormats_BTauReco_HLTOpenBJet_h

#include <vector>

namespace trigger {

struct HLTOpenBJetJetProperties {
  HLTOpenBJetJetProperties() :
    m_jetEnergy(),
    m_btagDiscriminantL25(),
    m_btagDiscriminantL3()
  { }

  HLTOpenBJetJetProperties(double energy, double l25, double l3) :
    m_jetEnergy( energy ),
    m_btagDiscriminantL25( l25 ),
    m_btagDiscriminantL3( l3 )
  { }

  double m_jetEnergy;
  double m_btagDiscriminantL25;
  double m_btagDiscriminantL3;
};

bool operator<(const HLTOpenBJetJetProperties & lhs, const HLTOpenBJetJetProperties & rhs) {
  return (lhs.m_jetEnergy < rhs.m_jetEnergy);
}


class HLTOpenBJet {
public:
  HLTOpenBJet() :
    m_hadronicEnergy(),
    m_properties() 
  { }

  void setHadronicEnergy(double energy) {
    m_hadronicEnergy = energy;
  }

  void addJet(double energy, double l25, double l3) {
    m_properties.push_back( HLTOpenBJetJetProperties(energy, l25, l3) );
    std::stable_sort(m_properties.begin(), m_properties.end());
  }

  /// return the total tranverse adronic energy
  double hadronicEnergy() const {
    return m_hadronicEnergy;
  }

  /// return the number of jets
  unsigned int jets() const {
    return m_properties.size();
  }

  /// return the energy of the n-th jet (counting from 0)
  double jetEnergy(unsigned int jet) {
    return (jet < m_properties.size()) ? m_properties[jet].m_jetEnergy : 0.;
  }

  /// return the L2.5 b-tagging discriminant of the n-th jet (counting from 0)
  double jetDiscriminantL25At(unsigned int jet) {
    return (jet < m_properties.size()) ? m_properties[jet].m_btagDiscriminantL25 : 0.;
  }
  
  /// return the L3 b-tagging discriminant of the n-th jet (counting from 0)
  double jetDiscriminantL3At(unsigned int jet) {
    return (jet < m_properties.size()) ? m_properties[jet].m_btagDiscriminantL3 : 0.;
  }
  
  /// return the highest jet L2.5 b-tagging discriminant, among the first n jets
  double jetDiscriminantL25(unsigned int jets) {
    if (jets > m_properties.size()) jets = m_properties.size();
    double tag = 0.;
    for (unsigned int i = 0; i < jets; ++i)
      if (m_properties[i].m_btagDiscriminantL25 > tag)
        tag = m_properties[i].m_btagDiscriminantL25;
    return tag;
  }
 
  /// return the highest jet L3 b-tagging discriminant, among the first n jets
  double jetDiscriminantL3(unsigned int jets) {
    if (jets > m_properties.size()) jets = m_properties.size();
    double tag = 0.;
    for (unsigned int i = 0; i < m_properties.size(); ++i)
      if (m_properties[i].m_btagDiscriminantL3 > tag)
        tag = m_properties[i].m_btagDiscriminantL3;
    return tag;
  }
  
  /// return the highest jet L2.5 b-tagging discriminant
  double jetDiscriminant25() {
    unsigned int jets = m_properties.size();
    double tag = 0.;
    for (unsigned int i = 0; i < jets; ++i)
      if (m_properties[i].m_btagDiscriminantL25 > tag)
        tag = m_properties[i].m_btagDiscriminantL25;
    return tag;
  }
 
  /// return the highest jet L3 b-tagging discriminant
  double jetDiscriminantL3() {
    unsigned int jets = m_properties.size();
    double tag = 0.;
    for (unsigned int i = 0; i < m_properties.size(); ++i)
      if (m_properties[i].m_btagDiscriminantL3 > tag)
        tag = m_properties[i].m_btagDiscriminantL3;
    return tag;
  }
  
private:
  double m_hadronicEnergy;
  std::vector<HLTOpenBJetJetProperties> m_properties;

};

}

#endif // DataFormats_BTauReco_HLTOpenBJet_h

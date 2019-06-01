#ifndef DataFormats_L1GlobalTrigger_L1GtTriggerMenuLite_h
#define DataFormats_L1GlobalTrigger_L1GtTriggerMenuLite_h

/**
 * \class L1GtTriggerMenuLite
 *
 *
 * Description: L1 trigger menu and masks, lite version not using event setup.
 *
 * Implementation:
 *    This is the lite version of the L1 trigger menu, with trigger masks included,
 *    to be used in the environments not having access to event setup. It offers
 *    limited access to the full L1 trigger menu which is implemented as event setup
 *    (CondFormats/L1TObjects/interface/L1GtTriggerMenu.h). The masks are provided for
 *    the physics partition only.
 *
 *    An EDM product is created and saved in the Run Data, under the assumption that the
 *    menu remains the same in a run.
 *    The corresponding producer will read the full L1 trigger menu and the trigger masks
 *    from event setup, fill the corresponding members and save it as EDM product.
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// system include files
#include <string>
#include <vector>
#include <map>

#include <iosfwd>

// user include files

// forward declarations

// class declaration
class L1GtTriggerMenuLite {
public:
  /// map containing the physics algorithms or the technical triggers
  typedef std::map<unsigned int, std::string> L1TriggerMap;

  /// iterators through map containing the physics algorithms or the technical triggers
  typedef L1TriggerMap::const_iterator CItL1Trig;
  typedef L1TriggerMap::iterator ItL1Trig;

public:
  /// constructor
  L1GtTriggerMenuLite();

  L1GtTriggerMenuLite(const std::string&,
                      const L1TriggerMap&,
                      const L1TriggerMap&,
                      const L1TriggerMap&,
                      const std::vector<unsigned int>&,
                      const std::vector<unsigned int>&,
                      const std::vector<std::vector<int> >&,
                      const std::vector<std::vector<int> >&);

  /// copy constructor
  L1GtTriggerMenuLite(const L1GtTriggerMenuLite&);

  // destructor
  virtual ~L1GtTriggerMenuLite();

  /// assignment operator
  L1GtTriggerMenuLite& operator=(const L1GtTriggerMenuLite&);

  /// equal operator
  bool operator==(const L1GtTriggerMenuLite&) const;

  /// unequal operator
  bool operator!=(const L1GtTriggerMenuLite&) const;

  /// merge rule: test on isProductEqual
  bool isProductEqual(const L1GtTriggerMenuLite&) const;

public:
  /// get / set the trigger menu names
  inline const std::string& gtTriggerMenuInterface() const { return m_triggerMenuInterface; }

  void setGtTriggerMenuInterface(const std::string&);

  //
  inline const std::string& gtTriggerMenuName() const { return m_triggerMenuName; }

  void setGtTriggerMenuName(const std::string&);

  //
  inline const std::string& gtTriggerMenuImplementation() const { return m_triggerMenuImplementation; }

  void setGtTriggerMenuImplementation(const std::string&);

  /// menu associated scale key
  inline const std::string& gtScaleDbKey() const { return m_scaleDbKey; }

  void setGtScaleDbKey(const std::string&);

  /// get / set the algorithm map (by name)
  inline const L1TriggerMap& gtAlgorithmMap() const { return m_algorithmMap; }

  void setGtAlgorithmMap(const L1TriggerMap&);

  /// get / set the algorithm map (by alias)
  inline const L1TriggerMap& gtAlgorithmAliasMap() const { return m_algorithmAliasMap; }

  void setGtAlgorithmAliasMap(const L1TriggerMap&);

  /// get / set the technical trigger map
  inline const L1TriggerMap& gtTechnicalTriggerMap() const { return m_technicalTriggerMap; }

  void setGtTechnicalTriggerMap(const L1TriggerMap&);

  /// get the trigger mask for physics algorithms
  inline const std::vector<unsigned int>& gtTriggerMaskAlgoTrig() const { return m_triggerMaskAlgoTrig; }

  /// set the trigger mask for physics algorithms
  void setGtTriggerMaskAlgoTrig(const std::vector<unsigned int>&);

  /// get the trigger mask for technical triggers
  inline const std::vector<unsigned int>& gtTriggerMaskTechTrig() const { return m_triggerMaskTechTrig; }

  /// set the trigger mask for technical triggers
  void setGtTriggerMaskTechTrig(const std::vector<unsigned int>&);

  /// get the prescale factors by reference / set the prescale factors
  inline const std::vector<std::vector<int> >& gtPrescaleFactorsAlgoTrig() const { return m_prescaleFactorsAlgoTrig; }

  void setGtPrescaleFactorsAlgoTrig(const std::vector<std::vector<int> >&);

  inline const std::vector<std::vector<int> >& gtPrescaleFactorsTechTrig() const { return m_prescaleFactorsTechTrig; }

  void setGtPrescaleFactorsTechTrig(const std::vector<std::vector<int> >&);

  /// print the trigger menu
  /// allow various verbosity levels
  void print(std::ostream&, int&) const;

  /// output stream operator
  friend std::ostream& operator<<(std::ostream&, const L1GtTriggerMenuLite&);

public:
  /// get the alias for a physics algorithm with a given bit number
  const std::string* gtAlgorithmAlias(const unsigned int bitNumber, int& errorCode) const;

  /// get the name for a physics algorithm or a technical trigger
  /// with a given bit number
  const std::string* gtAlgorithmName(const unsigned int bitNumber, int& errorCode) const;
  const std::string* gtTechTrigName(const unsigned int bitNumber, int& errorCode) const;

  /// get the bit number for a physics algorithm or a technical trigger
  /// with a given name or alias
  const unsigned int gtBitNumber(const std::string& trigName, int& errorCode) const;

  /// get the result for a physics algorithm or a technical trigger with name trigName
  /// use directly the format of decisionWord (no typedef)
  const bool gtTriggerResult(const std::string& trigName, const std::vector<bool>& decWord, int& errorCode) const;

private:
  /// menu names
  std::string m_triggerMenuInterface;
  std::string m_triggerMenuName;
  std::string m_triggerMenuImplementation;

  /// menu associated scale key
  std::string m_scaleDbKey;

  /// map containing the physics algorithms (by name)
  L1TriggerMap m_algorithmMap;

  /// map containing the physics algorithms (by alias)
  L1TriggerMap m_algorithmAliasMap;

  /// map containing the technical triggers
  L1TriggerMap m_technicalTriggerMap;

  /// trigger mask for physics algorithms
  std::vector<unsigned int> m_triggerMaskAlgoTrig;

  /// trigger mask for technical triggers
  std::vector<unsigned int> m_triggerMaskTechTrig;

  /// prescale factors
  std::vector<std::vector<int> > m_prescaleFactorsAlgoTrig;
  std::vector<std::vector<int> > m_prescaleFactorsTechTrig;
};

#endif /*DataFormats_L1GlobalTrigger_L1GtTriggerMenuLite_h*/

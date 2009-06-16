#ifndef SiStripApvGain_h
#define SiStripApvGain_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

/**
 * Stores the information of the gain for each apv using four vectors <br>
 * A vector<unsigned int> (v_detids) stores the detId. <br>
 * A vector<float> (v_gains) stores the value of the gain (more than one per detId). <br>
 * Two vector<unsigned int> (v_ibegin and v_iend) store the correspondence of the v_detids
 * and the ranges of values in v_gain. <br>
 *
 * The printSummary method uses SiStripDetSummary. See description therein. <br>
 * The printDebug method prints the gain value for every apv of every detId. <br>
 */

class SiStripApvGain {

	public:

  typedef std::vector<float>::const_iterator               ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<unsigned int>                        Registry;
  typedef Registry::iterator                               RegistryIterator;
  typedef Registry::const_iterator                         RegistryConstIterator;
  typedef std::vector<float>                               InputVector;

  SiStripApvGain(){};
  ~SiStripApvGain(){};
  
  bool put(const uint32_t& detID, Range input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  float   getStripGain     (const uint16_t& strip, const Range& range) const;
  float   getApvGain  (const uint16_t& apv, const Range& range) const;

  void printDebug(std::stringstream & ss) const;
  void printSummary(std::stringstream & ss) const;

 private:

  std::vector<float> v_gains;
  std::vector<unsigned int>   v_detids;
  std::vector<unsigned int>   v_ibegin;
  std::vector<unsigned int>   v_iend;
};

#endif

#ifndef _PATTERNTREE_H_
#define _PATTERNTREE_H_

#include <map>
#include <vector>
#include "PatternTrunk.h"

using namespace std;
/**
   \brief This class is used to store all the patterns (both low and high resolutions). To quickly find a pattern, PatternTrunk objects are stored in a map using the low resolution pattern value as the map key (string made of the decimal representation of the patternLayers values).
**/
class PatternTree{
 public:
  /**
     \brief Constructor
  **/
  PatternTree();
  ~PatternTree();
 /**
     \brief Add a copy of the pattern to the structure or increment the grade of the already stored pattern
     \param ldp The low definition pattern (will be copied)
     \param fdp The corresponding full definition pattern (will be copied, can be NULL if only one definition level is used)
  **/
  void addPattern(Pattern* ldp, Pattern* fdp);
 /**
     \brief Add a pattern to the structure or increment the grade of the already stored pattern. Update the average Pt of the pattern.
     \param ldp The low definition pattern
     \param fdp The corresponding full definition pattern (can be NULL if only one definition level is used)
     \param new_pt The Pt of the track generating the pattern
  **/
  void addPattern(Pattern* ldp, Pattern* fdp, float new_pt);

 /**
     \brief Get a copy of the full definition patterns
     \return A vector of copies
  **/
  vector<GradedPattern*> getFDPatterns();
 /**
     \brief Get a copy of the low definition patterns
     \return A vector of copies
  **/
  vector<GradedPattern*> getLDPatterns();
  /**
     \brief Get the distribution of PT among the LDPatterns
     \return A vector containing the occurences of each PT (bin is 1).
  **/
  vector<int> getPTHisto();
  /**
     \brief Get the number of LD patterns
     \return The number of LD patterns in the structure
  **/
  int getLDPatternNumber();
  /**
     \brief Get the number of FD patterns
     \return The number of FD patterns in the structure
  **/
  int getFDPatternNumber();
  /**
     \brief Link all patterns to the detector structure
     \param d The detector
     \param sec The ladders in the sector (one vector per layer)
     \param modules The modules in the sector (one vector per ladder)
  **/
  void link(Detector& d, const vector< vector<int> >& sec, const vector<map<int, vector<int> > >& modules);
  /**
     \brief Returns a vector of copies of the active patterns
     \brief active_threshold The minimum number of hit super strips to activate the pattern
     \return A vector containing copies of active patterns
  **/
  void getActivePatterns(int active_threshold, vector<GradedPattern*>& active_patterns);
  /**
     \brief Replace all LD patterns with adapatative patterns. All FD patterns are removed.
     \param r The number of DC bits used between FD and LD
  **/
  void computeAdaptativePatterns(short r);
  /**
     \brief Add all LD patterns coming from an other PatternTree
     \param p The PatternTree containing the patterns to add
  **/
  void addPatternsFromTree(PatternTree* p);
  
  /**
     \brief Check if the given pattern is contained in the bank (using DC bits)
     \brief Should only be used with a DC bit activated bank
     \param lp The low definition pattern
     \param hp The high definition version of the pattern
     \return True if the pattern is already in the bank, false otherwise
   **/
  bool checkPattern(Pattern* lp, Pattern* hp);

 private:
  map<string, PatternTrunk*> patterns;

  /**
     \brief Add a pattern and update de DC bits if necessary
     \param ldp The pattern to add
  **/
  void addPatternForMerging(GradedPattern* ldp);

  friend class boost::serialization::access;
 
  template<class Archive> void save(Archive & ar, const unsigned int version) const{
    ar << patterns;
  }
  
  template<class Archive> void load(Archive & ar, const unsigned int version){
    ar >> patterns;
  }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};
#endif

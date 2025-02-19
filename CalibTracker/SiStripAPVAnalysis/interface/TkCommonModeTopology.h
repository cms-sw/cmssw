#ifndef TkCommonModeTopology_h
#define TkCommonModeTopology_h

#include <vector>
/**
 * Allows any topology for the Common Mode: 128 strips, 64, 32, 16, 8, ....
 */
class TkCommonModeTopology{
 public:
  
  TkCommonModeTopology(int nstrips, int nstripsperset);
  
  /** Set number of strips in an APV = 128 */
  void setNumberOfStrips(int in) {numberStrips = in;} 
  /** Set number of strips in each group for which CM is to be found */
  void setNumberOfStripsPerSet(int in) {numberStripsPerSet = in;}   
  /** Set number of independent groups of strips in APV for CM */
  void setNumberOfSets (int in) {numberStripsPerSet = numberStrips/in;}
  
  int numberOfStrips() const {return numberStrips;}
  int numberOfStripsPerSet() const {return numberStripsPerSet;}

  int numberOfSets() const {return numberStrips/numberStripsPerSet;}

  int setOfStrip(int);
  
  std::vector<int>& initialStrips(){return initStrips;}
  std::vector<int>& finalStrips(){return finStrips;}
  
 private:
  int numberStrips;
  int numberStripsPerSet;
  std::vector<int>initStrips;
  std::vector<int>finStrips;
};

#endif
